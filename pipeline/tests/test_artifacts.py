"""Tests fuer den Datenvertrag.

Der Schwerpunkt liegt auf den beiden Invarianten, deren Verletzung sonst erst im
Browser auffaellt — und dort als stille Fehlzuordnung zwischen Punkt und
Metadatum, nicht als Absturz.
"""

from __future__ import annotations

import numpy as np
import pytest

from lse import artifacts, config as config_module, synthetic


@pytest.fixture()
def cfg():
    return config_module.load()


@pytest.fixture()
def run_dir(tmp_path):
    return tmp_path / "run"


# ---------------------------------------------------------------------------
# Koordinaten
# ---------------------------------------------------------------------------


def test_normalize_is_uniform_and_preserves_shape():
    """Skalierung pro Achse wuerde die Projektion verzerren — hier nachgewiesen."""
    rng = np.random.default_rng(0)
    # Bewusst stark anisotrop: eine Achse zehnmal so breit wie die anderen.
    coords = rng.normal(size=(500, 3)) * np.array([10.0, 1.0, 1.0])

    normalized, centroid, scale = artifacts.normalize_coords(coords, box=50.0)

    # Der Seitenverhaeltnis-Test: Verhaeltnisse der Achsenausdehnungen bleiben.
    # np.ptp als Funktion, nicht als Methode — die Methode ist in numpy 2.0
    # entfallen.
    before = np.ptp(coords, axis=0)
    after = np.ptp(normalized, axis=0)
    np.testing.assert_allclose(after / after[0], before / before[0], rtol=1e-9)

    assert np.abs(normalized).max() == pytest.approx(50.0)
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(centroid, coords.mean(axis=0))
    assert scale > 0


def test_quantize_roundtrip_within_resolution():
    """int16 ueber die Box +/-50 muss deutlich feiner sein, als das Auge braucht."""
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(2000, 3)) * 12.0

    normalized, _, _ = artifacts.normalize_coords(coords, box=50.0)
    q, spec = artifacts.prepare_coords(coords, box=50.0)

    restored = artifacts.dequantize_coords(q, spec)
    error = np.abs(restored - normalized).max()

    # Halbe Quantisierungsstufe der breitesten Achse.
    step = max(spec.half) / artifacts.I16_SCALE
    assert error <= step, f"Fehler {error} ueber der Stufe {step}"
    assert error < 0.01, "Aufloesung reicht nicht fuer eine Box von +/-50"


def test_quantize_handles_degenerate_axis():
    """Eine flache Wolke darf keine Division durch null ausloesen."""
    coords = np.zeros((100, 3))
    coords[:, 0] = np.linspace(-1, 1, 100)
    coords[:, 1] = np.linspace(-1, 1, 100)
    # z ist konstant.

    q, spec = artifacts.prepare_coords(coords, box=50.0)
    restored = artifacts.dequantize_coords(q, spec)

    assert np.isfinite(restored).all()
    np.testing.assert_allclose(restored[:, 2], 0.0, atol=1e-9)


def test_identical_points_do_not_crash():
    coords = np.full((10, 3), 3.5)
    q, spec = artifacts.prepare_coords(coords, box=50.0)
    assert np.isfinite(artifacts.dequantize_coords(q, spec)).all()
    assert spec.radius == pytest.approx(0.0)


def test_rejects_wrong_shape():
    with pytest.raises(artifacts.ArtifactError):
        artifacts.normalize_coords(np.zeros((10, 2)), box=50.0)
    with pytest.raises(artifacts.ArtifactError):
        artifacts.normalize_coords(np.zeros((0, 3)), box=50.0)


# ---------------------------------------------------------------------------
# Binaerdateien
# ---------------------------------------------------------------------------


def test_binary_roundtrip_is_little_endian(tmp_path):
    arr = np.arange(30, dtype="<i2").reshape(10, 3)
    path = tmp_path / "coords.i16.bin"

    entry = artifacts.write_binary(path, arr, "<i2")
    assert entry.bytes == 60
    assert entry.shape == [10, 3]

    # Byteweise gegen die erwartete Little-Endian-Darstellung.
    assert path.read_bytes()[:4] == b"\x00\x00\x01\x00"

    back = artifacts.read_binary(path, "<i2", (10, 3))
    np.testing.assert_array_equal(back, arr)


def test_read_binary_detects_truncation(tmp_path):
    path = tmp_path / "x.bin"
    artifacts.write_binary(path, np.arange(30, dtype="<i2").reshape(10, 3), "<i2")
    with pytest.raises(artifacts.ArtifactError, match="erwartet"):
        artifacts.read_binary(path, "<i2", (11, 3))


# ---------------------------------------------------------------------------
# ID-Invariante
# ---------------------------------------------------------------------------


def test_finish_rejects_mismatched_row_counts(cfg, run_dir):
    """Der Fehler, der sonst still zu falschen Metadaten am Punkt fuehrt."""
    coords, labels, names = synthetic.make_coords(100, seed=0)

    writer = artifacts.ArtifactWriter(run_dir, cfg)
    writer.add_coords(coords)
    writer.add_labels(labels[:99], names)  # eine Zeile zu wenig

    with pytest.raises(artifacts.ArtifactError, match="ID-Invariante"):
        writer.finish()


def test_finish_requires_coords(cfg, run_dir):
    writer = artifacts.ArtifactWriter(run_dir, cfg)
    with pytest.raises(artifacts.ArtifactError, match="add_coords"):
        writer.finish()


def test_labels_need_names(cfg, run_dir):
    coords, labels, _ = synthetic.make_coords(50, n_clusters=4, seed=0)
    writer = artifacts.ArtifactWriter(run_dir, cfg)
    writer.add_coords(coords)
    with pytest.raises(artifacts.ArtifactError, match="keinen Namen"):
        writer.add_labels(labels, ["nur-einer"])


# ---------------------------------------------------------------------------
# Vollstaendiger Satz
# ---------------------------------------------------------------------------


def _write_set(cfg, run_dir, n=200):
    coords, labels, names = synthetic.make_coords(n, seed=3)
    writer = artifacts.ArtifactWriter(run_dir, cfg)
    writer.add_coords(coords)
    writer.add_labels(labels, names)
    writer.add_colors(synthetic.palette_colors(labels))
    writer.add_neighbors(np.zeros((n, 4), dtype=np.uint32))
    return writer.finish(corpus={"id": "test"}), coords, labels


def test_full_set_verifies(cfg, run_dir):
    manifest, _, _ = _write_set(cfg, run_dir)

    assert manifest.n == 200
    assert manifest.schema_version == config_module.SCHEMA_VERSION
    assert artifacts.verify(run_dir) == []

    # Das Manifest muss reproduzierbar machen, was den Lauf erzeugt hat.
    assert "libraries" in manifest.provenance
    assert "numpy" in manifest.provenance["libraries"]
    assert manifest.provenance["config"]["schema_version"] == config_module.SCHEMA_VERSION


def test_verify_detects_corruption(cfg, run_dir):
    _write_set(cfg, run_dir)
    path = run_dir / artifacts.COORDS_FILE

    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF  # ein Bit kippen, Groesse unveraendert
    path.write_bytes(bytes(data))

    problems = artifacts.verify(run_dir)
    assert any("Pruefsumme" in p for p in problems)


def test_verify_detects_missing_file(cfg, run_dir):
    _write_set(cfg, run_dir)
    (run_dir / artifacts.COLORS_FILE).unlink()

    problems = artifacts.verify(run_dir)
    assert any("fehlt" in p for p in problems)


def test_coords_survive_the_full_write_read_cycle(cfg, run_dir):
    """Der eigentliche Vertrag: was der Viewer liest, ist was die Pipeline meinte."""
    manifest, coords, _ = _write_set(cfg, run_dir, n=500)

    spec = artifacts.CoordSpec(
        offset=tuple(manifest.coords["offset"]),
        half=tuple(manifest.coords["half"]),
        min=tuple(manifest.coords["min"]),
        max=tuple(manifest.coords["max"]),
        radius=manifest.coords["radius"],
        source_centroid=tuple(manifest.coords["source_centroid"]),
        source_scale=manifest.coords["source_scale"],
        box=manifest.coords["box"],
    )

    stride = manifest.coords["coord_stride"]
    raw = artifacts.read_binary(run_dir / artifacts.COORDS_FILE, "<i2", (manifest.n, stride))
    restored = artifacts.dequantize_coords(raw, spec)

    expected, _, _ = artifacts.normalize_coords(coords, box=spec.box)
    np.testing.assert_allclose(restored, expected, atol=0.01)


def test_vertex_attributes_are_four_wide(cfg, run_dir):
    """WebGPU kennt keine dreikomponentigen 8/16-Bit-Vertexformate.

    Ohne die Auffuellung muesste der Browser beim Laden einmal ueber alle Punkte
    umkopieren — genau die Arbeit, die der binaere Vertrag vermeiden soll.
    """
    manifest, _, _ = _write_set(cfg, run_dir, n=64)

    by_name = {f["name"]: f for f in manifest.files}
    assert by_name[artifacts.COORDS_FILE]["shape"] == [64, 4]
    assert by_name[artifacts.COLORS_FILE]["shape"] == [64, 4]
    assert manifest.coords["coord_stride"] == 4
    assert manifest.coords["color_stride"] == 4

    # Die Groessen muessen exakt aufgehen, sonst liest der Browser versetzt.
    assert by_name[artifacts.COORDS_FILE]["bytes"] == 64 * 4 * 2
    assert by_name[artifacts.COLORS_FILE]["bytes"] == 64 * 4 * 1

    # Vierte Koordinatenkomponente ist reserviert, vierte Farbkomponente opak.
    raw_coords = artifacts.read_binary(run_dir / artifacts.COORDS_FILE, "<i2", (64, 4))
    assert (raw_coords[:, 3] == 0).all()
    raw_colors = artifacts.read_binary(run_dir / artifacts.COLORS_FILE, "<u1", (64, 4))
    assert (raw_colors[:, 3] == 255).all()


def test_manifest_roundtrip(cfg, run_dir):
    manifest, _, _ = _write_set(cfg, run_dir)
    again = artifacts.load_manifest(run_dir)
    assert again.n == manifest.n
    assert again.corpus["id"] == "test"
    assert len(again.files) == len(manifest.files)


# ---------------------------------------------------------------------------
# Metadaten
# ---------------------------------------------------------------------------


def _rows(n=50):
    return [
        {
            "title": f"Werk {i}",
            "artist": "Anonym" if i % 3 else "",
            "department": ["Egyptian Art", "Asian Art"][i % 2],
            "culture": ["Egyptian", "Japan", "unbekannt"][i % 3],
            "link": f"http://example.org/{i}",
        }
        for i in range(n)
    ]


def _write_with_metadata(cfg, run_dir, rows):
    coords, labels, names = synthetic.make_coords(len(rows), seed=1)
    writer = artifacts.ArtifactWriter(run_dir, cfg)
    writer.add_coords(coords)
    writer.add_labels(labels, names)
    writer.add_colors(synthetic.palette_colors(labels))
    meta = writer.add_metadata(
        rows,
        facet_fields=["department", "culture"],
        text_fields=["title", "artist", "link"],
    )
    return writer.finish(corpus={"id": "test"}), meta


def test_metadata_facets_and_text(cfg, run_dir):
    rows = _rows(50)
    manifest, meta = _write_with_metadata(cfg, run_dir, rows)

    assert meta["facet_fields"] == ["department", "culture"]
    assert meta["facet_values"]["department"] == ["Asian Art", "Egyptian Art"]
    assert artifacts.verify(run_dir) == []
    assert manifest.metadata["text_fields"] == ["title", "artist", "link"]

    facets = artifacts.read_binary(run_dir / artifacts.FACETS_FILE, "<u1", (50, 2))
    vocabulary = meta["facet_values"]["department"]
    for row, source in enumerate(rows):
        assert vocabulary[facets[row, 0]] == source["department"]


def test_text_offsets_have_n_plus_one_entries(cfg, run_dir):
    """Damit die Laenge des letzten Datensatzes ohne Sonderfall ableitbar ist."""
    rows = _rows(20)
    manifest, _ = _write_with_metadata(cfg, run_dir, rows)

    offsets = artifacts.read_binary(run_dir / artifacts.TEXT_OFFSETS_FILE, "<u4", (21,))
    assert offsets[0] == 0
    assert offsets[-1] == (run_dir / artifacts.TEXT_FILE).stat().st_size
    assert (np.diff(offsets) > 0).all()
    # verify() darf die N+1-Zeile nicht als Invariantenbruch melden.
    assert artifacts.verify(run_dir) == []


def test_text_roundtrips_including_umlauts_and_empty_fields(cfg, run_dir):
    rows = [
        {"title": "Löwenkopf aus Ägypten", "artist": "", "link": "http://x/1"},
        {"title": "祭器", "artist": "無名", "link": ""},
        {"title": "Œuvre — mit Gedankenstrich", "artist": "A|B;C", "link": "http://x/3"},
    ] * 10
    manifest, meta = _write_with_metadata(
        cfg, run_dir, [{**row, "department": "X", "culture": "Y"} for row in rows]
    )

    blob = (run_dir / artifacts.TEXT_FILE).read_bytes()
    offsets = artifacts.read_binary(
        run_dir / artifacts.TEXT_OFFSETS_FILE, "<u4", (len(rows) + 1,)
    )
    for index, source in enumerate(rows):
        record = blob[offsets[index] : offsets[index + 1]].decode("utf-8")
        parts = record.split(artifacts.FIELD_SEPARATOR)
        assert parts[0] == source["title"]
        assert parts[1] == source["artist"]
        assert parts[2] == source["link"]


def test_separator_cannot_collide_with_real_metadata(cfg, run_dir):
    """ASCII 31 ist dafuer gedacht — anders als Pipe, Semikolon oder Tab.

    Alle drei kommen in echten Met-Titeln und -Materialangaben vor.
    """
    rows = [
        {
            "title": "Teil A|Teil B; Teil C\tTeil D",
            "artist": "X",
            "link": "",
            "department": "X",
            "culture": "Y",
        }
    ] * 5
    _write_with_metadata(cfg, run_dir, rows)

    blob = (run_dir / artifacts.TEXT_FILE).read_bytes()
    offsets = artifacts.read_binary(run_dir / artifacts.TEXT_OFFSETS_FILE, "<u4", (6,))
    record = blob[offsets[0] : offsets[1]].decode("utf-8")
    assert record.split(artifacts.FIELD_SEPARATOR)[0] == "Teil A|Teil B; Teil C\tTeil D"


def test_metadata_participates_in_id_invariant(cfg, run_dir):
    """Zu wenige Metadatenzeilen muessen auffallen, nicht still verrutschen."""
    coords, labels, names = synthetic.make_coords(30, seed=2)
    writer = artifacts.ArtifactWriter(run_dir, cfg)
    writer.add_coords(coords)
    writer.add_labels(labels, names)
    writer.add_metadata(
        _rows(29), facet_fields=["department"], text_fields=["title"]
    )
    with pytest.raises(artifacts.ArtifactError, match="ID-Invariante"):
        writer.finish()


def test_facet_with_too_many_values_is_rejected(cfg, run_dir):
    """uint8 traegt 256 Werte — darueber muss im Korpusschritt gekappt werden."""
    rows = [
        {"title": f"T{i}", "department": f"Wert-{i}", "culture": "X"} for i in range(300)
    ]
    coords, labels, names = synthetic.make_coords(300, seed=3)
    writer = artifacts.ArtifactWriter(run_dir, cfg)
    writer.add_coords(coords)
    writer.add_labels(labels, names)
    with pytest.raises(artifacts.ArtifactError, match="cap_facet"):
        writer.add_metadata(rows, facet_fields=["department"], text_fields=["title"])


# ---------------------------------------------------------------------------
# Synthetische Daten
# ---------------------------------------------------------------------------


def test_high_dim_neighbors_are_sane():
    embeddings, labels, _ = synthetic.make_embeddings(300, dim=32, n_clusters=4, seed=7)
    neighbors = synthetic.high_dim_neighbors(embeddings, k=5)

    assert neighbors.shape == (300, 5)
    assert neighbors.dtype == np.uint32
    # Niemand ist sein eigener Nachbar.
    assert not (neighbors == np.arange(300)[:, None]).any()

    # Bei gut getrennten Blobs muessen die Nachbarn ueberwiegend dasselbe Label
    # tragen — sonst misst der Rest der Pipeline Rauschen.
    same = (labels[neighbors] == labels[:, None]).mean()
    assert same > 0.9, f"nur {same:.2f} der Nachbarn teilen das Label"


@pytest.mark.parametrize("dim", [16, 128, 768])
def test_cluster_separation_is_dimension_independent(dim):
    """``spread`` muss in jeder Dimension dasselbe bedeuten.

    Regressionstest fuer einen echten Fehler: als ``spread`` noch die Streuung
    *pro Dimension* war, hatte der Rauschvektor bei dim=768 die Norm ~9.7,
    waehrend zwei zufaellige Einheitszentren nur ~1.41 auseinanderliegen. Der
    Generator behauptete Cluster und lieferte Rauschen — und haette den
    Renderer- und Validierungspfad gegen eine Illusion getestet.
    """
    embeddings, labels, _ = synthetic.make_embeddings(
        400, dim=dim, n_clusters=4, seed=11, l2_normalize=False
    )
    neighbors = synthetic.high_dim_neighbors(embeddings, k=5)
    same = (labels[neighbors] == labels[:, None]).mean()
    assert same > 0.95, f"dim={dim}: nur {same:.2f} der Nachbarn teilen das Label"


def test_spread_actually_controls_separation():
    """``spread`` muss die Trennbarkeit monoton steuern — sonst ist der Regler tot.

    Geprueft wird die Kurve, nicht ein einzelner Schwellwert: bei vier Clustern
    liegt das Zufallsniveau bei 0,25, und die Reinheit muss von "perfekt" dorthin
    absinken. Gemessen (dim=64, k=5): 0,2 -> 1,00 | 1,0 -> 1,00 | 2,0 -> 0,93 |
    3,0 -> 0,68 | 5,0 -> 0,43 | 10,0 -> 0,29.
    """
    spreads = (0.2, 2.0, 5.0, 10.0)
    purities = []
    for spread in spreads:
        embeddings, labels, _ = synthetic.make_embeddings(
            400, dim=64, n_clusters=4, seed=5, spread=spread, l2_normalize=False
        )
        neighbors = synthetic.high_dim_neighbors(embeddings, k=5)
        purities.append(float((labels[neighbors] == labels[:, None]).mean()))

    assert purities[0] > 0.95, "enge Cluster muessen sauber trennbar sein"
    assert all(a > b for a, b in zip(purities, purities[1:])), (
        f"Reinheit muss mit steigendem spread monoton fallen, ist aber {purities}"
    )
    # Zufallsniveau ist 1/4; bei spread=10 muss der Rest davon nahezu weg sein.
    assert purities[-1] < 0.40, f"bei spread=10 noch {purities[-1]:.2f} Reinheit"


def test_palette_is_colorblind_safe_length():
    labels = np.arange(20)
    colors = synthetic.palette_colors(labels)
    assert colors.shape == (20, 3)
    assert colors.dtype == np.uint8
    # Zyklisch, nicht abstuerzend, bei mehr Klassen als Farben.
    np.testing.assert_array_equal(colors[0], colors[len(synthetic.PALETTE)])
