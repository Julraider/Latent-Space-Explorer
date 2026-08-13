"""Tests fuer die Prompt-Platzierung und den lokalen Dienst."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from lse import artifacts, config as config_module, placement, server, synthetic


@pytest.fixture()
def cfg():
    return config_module.load()


# ---------------------------------------------------------------------------
# Platzierung
# ---------------------------------------------------------------------------


def _corpus(n=400, dim=32, clusters=4, seed=3):
    embeddings, labels, _ = synthetic.make_embeddings(
        n, dim=dim, n_clusters=clusters, seed=seed, spread=0.2, l2_normalize=False
    )
    # 3D-Koordinaten, die die Clusterstruktur tragen.
    from sklearn.decomposition import PCA

    coords = PCA(n_components=3, random_state=0).fit_transform(embeddings)
    return embeddings, coords, labels


def test_prompt_lands_among_its_own_cluster():
    """Der Kern: eine Anfrage nahe an Cluster X muss bei Cluster X landen."""
    embeddings, coords, labels = _corpus()

    # Anfrage = ein leicht verrauschtes Mitglied von Cluster 2.
    members = np.flatnonzero(labels == 2)
    rng = np.random.default_rng(0)
    query = embeddings[members[0]] + rng.normal(scale=0.05, size=embeddings.shape[1])

    result = placement.place(query, embeddings, coords, k=10)

    assert len(result.neighbors) == 10
    # Fast alle Nachbarn sollten aus demselben Cluster stammen.
    same = np.mean(labels[result.neighbors] == 2)
    assert same >= 0.9, f"nur {same:.2f} der Nachbarn aus dem richtigen Cluster"

    # Und die Position muss beim Schwerpunkt dieses Clusters liegen, nicht
    # irgendwo dazwischen.
    center = coords[members].mean(axis=0)
    spread_of_cluster = np.linalg.norm(coords[members] - center, axis=1).mean()
    assert np.linalg.norm(np.array(result.position) - center) < spread_of_cluster


def test_position_is_inside_the_hull_of_its_neighbors():
    """Ein gewichteter Schwerpunkt kann die Nachbarn nicht verlassen."""
    embeddings, coords, _ = _corpus()
    rng = np.random.default_rng(1)
    result = placement.place(rng.normal(size=32), embeddings, coords, k=8)

    picked = coords[result.neighbors]
    position = np.array(result.position)
    assert (position >= picked.min(axis=0) - 1e-9).all()
    assert (position <= picked.max(axis=0) + 1e-9).all()


def test_weights_sum_to_one_and_favour_better_matches():
    embeddings, coords, _ = _corpus()
    result = placement.place(embeddings[7], embeddings, coords, k=10)

    weights = np.array(result.weights)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights > 0).all()
    # Aehnlichkeiten sind absteigend sortiert, die Gewichte muessen folgen.
    assert (np.diff(weights) <= 1e-12).all()
    assert weights[0] > weights[-1]


def test_similarities_are_descending():
    embeddings, coords, _ = _corpus()
    result = placement.place(embeddings[3], embeddings, coords, k=12)
    assert (np.diff(result.similarities) <= 1e-6).all()


def test_softmax_is_scale_invariant():
    """Wegen des Modality Gap liegen Text-Bild-Werte in einem anderen Bereich.

    Eine feste Temperatur waere fuer einen der beiden Faelle sinnlos; die
    Gewichte muessen deshalb von der absoluten Lage der Aehnlichkeiten
    unabhaengig sein und nur von ihrer Spannweite abhaengen.
    """
    high = np.array([0.90, 0.88, 0.85, 0.80])
    low = high - 0.65  # dieselbe Spannweite, andere absolute Lage
    np.testing.assert_allclose(
        placement.softmax_weights(high), placement.softmax_weights(low), rtol=1e-9
    )


def test_spread_flags_an_ambiguous_prompt():
    """Nachbarn aus verschiedenen Ecken -> der Schwerpunkt liegt im Nichts.

    Der Korpus umfasst genau ``k`` Elemente, damit beide Gruppen zwingend in
    der Auswahl landen. Bei mehr Elementen und exakt gleichen Aehnlichkeiten
    entscheidet sonst die Indexreihenfolge, welche Gruppe genommen wird — dann
    misst der Test die Tie-Break-Regel statt der Streuung.
    """
    coords = np.vstack([
        np.random.default_rng(0).normal(loc=[-40, 0, 0], scale=1.0, size=(10, 3)),
        np.random.default_rng(1).normal(loc=[40, 0, 0], scale=1.0, size=(10, 3)),
    ])
    embeddings = np.vstack([
        np.tile([1.0, 0.0], (10, 1)),
        np.tile([0.0, 1.0], (10, 1)),
    ]).astype(np.float32)

    result = placement.place(
        np.array([1.0, 1.0], dtype=np.float32), embeddings, coords, k=20, radius=45.0
    )
    assert len(result.neighbors) == 20
    # Der Schwerpunkt landet zwischen den Gruppen, wo kein einziger Punkt liegt.
    assert abs(result.position[0]) < 5
    assert not result.confident
    assert "Leerstelle" in result.note
    assert result.spread > 0.55


def test_spread_is_low_for_a_coherent_prompt():
    """Gegenprobe: liegen alle Nachbarn beieinander, ist die Streuung klein."""
    coords = np.random.default_rng(2).normal(loc=[10, 10, 10], scale=1.0, size=(30, 3))
    embeddings = np.tile([1.0, 0.0], (30, 1)).astype(np.float32)

    result = placement.place(
        np.array([1.0, 0.05], dtype=np.float32), embeddings, coords, k=15, radius=45.0
    )
    assert result.confident
    assert result.spread < 0.1


def test_confident_prompt_has_no_note():
    embeddings, coords, _ = _corpus()
    result = placement.place(embeddings[11], embeddings, coords, k=8)
    assert result.confident
    assert result.note == ""


def test_rejects_mismatched_lengths():
    embeddings, coords, _ = _corpus(n=100)
    with pytest.raises(ValueError, match="ID-Invariante"):
        placement.place(embeddings[0], embeddings, coords[:99], k=5)


def test_k_larger_than_corpus_is_clamped():
    embeddings, coords, _ = _corpus(n=20)
    result = placement.place(embeddings[0], embeddings, coords, k=100)
    assert len(result.neighbors) == 20


# ---------------------------------------------------------------------------
# Stub-Encoder
# ---------------------------------------------------------------------------


def test_stub_is_deterministic_but_meaningless():
    encode = server.stub_encoder(64)
    first = encode("ein Boot bei Sonnenuntergang")
    again = encode("  Ein Boot bei Sonnenuntergang  ")  # normalisiert
    other = encode("ein Schiff bei Sonnenuntergang")

    np.testing.assert_array_equal(first, again)
    # Aehnliche Texte geben KEINE aehnlichen Vektoren — das ist der Punkt, und
    # deshalb ist der Stub ein Integrationstest, kein Feature.
    cosine = float(
        first @ other / (np.linalg.norm(first) * np.linalg.norm(other))
    )
    assert abs(cosine) < 0.4
    assert first.shape == (64,)


# ---------------------------------------------------------------------------
# Dienst
# ---------------------------------------------------------------------------


@pytest.fixture()
def running_service(cfg, tmp_path):
    """Ein echter HTTP-Server auf einem freien Port."""
    run_dir = tmp_path / "run"
    embeddings, labels, names = synthetic.make_embeddings(
        200, dim=16, n_clusters=4, seed=5, spread=0.2
    )
    coords, _, _ = synthetic.make_coords(200, n_clusters=4, seed=5)

    writer = artifacts.ArtifactWriter(run_dir, cfg)
    writer.add_coords(coords)
    writer.add_labels(labels, names)
    writer.add_colors(synthetic.palette_colors(labels))
    writer.add_embeddings(embeddings)
    writer.finish(corpus={"id": "test"})

    httpd = server.serve(run_dir, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, json.loads(response.read())


def _post(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health_reports_stub_and_size(running_service):
    status, payload = _get(f"{running_service}/health")
    assert status == 200
    assert payload["status"] == "ok"
    assert payload["stub"] is True
    assert payload["n"] == 200
    assert payload["dim"] == 16


def test_prompt_returns_a_placement(running_service):
    status, payload = _post(f"{running_service}/prompt", {"text": "eine Vase"})
    assert status == 200
    assert len(payload["position"]) == 3
    assert all(np.isfinite(payload["position"]))
    assert len(payload["neighbors"]) == 12
    assert payload["stub"] is True
    assert all(0 <= index < 200 for index in payload["neighbors"])


def test_same_prompt_gives_same_position(running_service):
    _, first = _post(f"{running_service}/prompt", {"text": "eine Vase"})
    _, again = _post(f"{running_service}/prompt", {"text": "eine Vase"})
    assert first["position"] == again["position"]


def test_empty_prompt_is_rejected(running_service):
    status, payload = _post(f"{running_service}/prompt", {"text": "   "})
    assert status == 400
    assert "error" in payload


def test_unknown_path_is_404(running_service):
    status, _ = _post(f"{running_service}/unbekannt", {"text": "x"})
    assert status == 404


def test_cors_header_is_present(running_service):
    """Der Viewer laeuft auf einem anderen Port — ohne CORS blockt der Browser."""
    with urllib.request.urlopen(f"{running_service}/health", timeout=10) as response:
        assert response.headers.get("Access-Control-Allow-Origin") == "*"


def test_service_without_embeddings_fails_clearly(cfg, tmp_path):
    run_dir = tmp_path / "leer"
    coords, labels, names = synthetic.make_coords(50, seed=1)
    writer = artifacts.ArtifactWriter(run_dir, cfg)
    writer.add_coords(coords)
    writer.add_labels(labels, names)
    writer.finish(corpus={"id": "test"})

    # Mit echtem Encoder ist das ein Fehler ...
    with pytest.raises(FileNotFoundError, match="keep-embeddings"):
        server.PromptService(run_dir, encoder=lambda text: np.zeros(4, dtype=np.float32))

    # ... im Stub-Modus dagegen faellt der Dienst auf die 3D-Koordinaten als
    # Suchraum zurueck, damit sich die Interaktion ohne Modell ausprobieren
    # laesst.
    service = server.PromptService(run_dir)
    assert service.space == "coords"
    assert service.dim == 3
    result = service.place("eine Vase")
    assert len(result["position"]) == 3
    assert result["stub"] is True
