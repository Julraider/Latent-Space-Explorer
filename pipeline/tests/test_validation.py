"""Tests fuer das Go/No-Go-Tor.

Der wichtigste Test ist :func:`test_shuffle_destroys_structure_row_shuffle_does_not`
— er haelt den Unterschied fest, an dem die ganze Kontrolle haengt.
"""

from __future__ import annotations

import numpy as np
import pytest

from lse import synthetic, validation


@pytest.fixture()
def blobs():
    return synthetic.make_embeddings(300, dim=32, n_clusters=4, seed=5, l2_normalize=False)


# ---------------------------------------------------------------------------
# Shuffle-Control
# ---------------------------------------------------------------------------


def test_shuffle_destroys_structure_row_shuffle_does_not(blobs):
    """Warum spaltenweise gemischt wird und nicht zeilenweise.

    Zeilen zu mischen ist dieselbe Punktmenge in anderer Reihenfolge — die
    Mannigfaltigkeit bleibt vollstaendig erhalten, und die "Kontrolle" wuerde
    exakt so gut aussehen wie die echten Daten. Nur eine unabhaengige
    Permutation je Spalte loest die Korrelationen zwischen den Dimensionen auf.
    """
    embeddings, labels, _ = blobs
    k = 10

    baseline = validation.label_purity(
        validation.knn_indices_high(embeddings, k), labels
    )

    rng = np.random.default_rng(0)
    # Zeilenpermutation: Punkte UND Labels wandern gemeinsam, sonst misst man
    # nicht die Struktur, sondern nur die kaputte Zuordnung.
    row_perm = rng.permutation(len(embeddings))
    row_purity = validation.label_purity(
        validation.knn_indices_high(embeddings[row_perm], k), labels[row_perm]
    )

    column_shuffled = validation.shuffle_columns(embeddings, seed=0)
    column_purity = validation.label_purity(
        validation.knn_indices_high(column_shuffled, k), labels
    )

    assert baseline > 0.95
    # Zeilenpermutation aendert nichts an der Struktur.
    assert row_purity == pytest.approx(baseline, abs=0.05)
    # Spaltenpermutation zerstoert sie.
    assert column_purity < 0.45, f"Spaltenmischung liess {column_purity:.2f} Reinheit uebrig"


def test_shuffle_preserves_marginal_distributions(blobs):
    """Jede Dimension behaelt ihre Verteilung — nur die Kombination faellt weg."""
    embeddings, _, _ = blobs
    shuffled = validation.shuffle_columns(embeddings, seed=1)

    np.testing.assert_allclose(shuffled.mean(axis=0), embeddings.mean(axis=0), rtol=1e-4)
    np.testing.assert_allclose(shuffled.std(axis=0), embeddings.std(axis=0), rtol=1e-4)
    for column in range(embeddings.shape[1]):
        np.testing.assert_allclose(
            np.sort(shuffled[:, column]), np.sort(embeddings[:, column]), rtol=1e-5
        )


def test_shuffle_does_not_mutate_input(blobs):
    embeddings, _, _ = blobs
    before = embeddings.copy()
    validation.shuffle_columns(embeddings, seed=2)
    np.testing.assert_array_equal(embeddings, before)


# ---------------------------------------------------------------------------
# Kennzahlen
# ---------------------------------------------------------------------------


def test_knn_excludes_self():
    coords = np.random.default_rng(0).normal(size=(50, 3))
    neighbors = validation.knn_indices_3d(coords, k=5)
    assert neighbors.shape == (50, 5)
    assert not (neighbors == np.arange(50)[:, None]).any()


def test_knn_overlap_bounds():
    high = np.tile(np.arange(5), (20, 1))
    assert validation.knn_overlap(high, high) == pytest.approx(1.0)

    disjoint = np.tile(np.arange(5, 10), (20, 1))
    assert validation.knn_overlap(high, disjoint) == pytest.approx(0.0)


def test_label_purity_bounds():
    labels = np.array([0, 0, 1, 1])
    same = np.array([[1], [0], [3], [2]])
    other = np.array([[2], [3], [0], [1]])
    assert validation.label_purity(same, labels) == pytest.approx(1.0)
    assert validation.label_purity(other, labels) == pytest.approx(0.0)


def test_chance_level_accounts_for_imbalance():
    """Bei unausgewogenen Klassen ist das Zufallsniveau nicht 1/Klassenzahl.

    Eine Klasse mit 90 % Anteil liefert schon zufaellig ~0,82 Reinheit — gegen
    1/2 = 0,5 zu vergleichen wuerde eine bedeutungslose Projektion durchwinken.
    """
    labels = np.array([0] * 90 + [1] * 10)
    coords = np.random.default_rng(0).normal(size=(100, 3))
    embeddings = np.random.default_rng(1).normal(size=(100, 8)).astype(np.float32)

    gate = validation.run_gate(embeddings, coords, labels, k=5)
    assert gate.label_purity_chance == pytest.approx(0.9**2 + 0.1**2)
    assert gate.label_purity_chance > 0.8


# ---------------------------------------------------------------------------
# Urteil
# ---------------------------------------------------------------------------


def test_gate_passes_on_real_structure():
    embeddings, labels, _ = synthetic.make_embeddings(
        400, dim=32, n_clusters=4, seed=9, spread=0.25, l2_normalize=False
    )
    # 3D-Koordinaten, die die Struktur tragen: die ersten drei PCA-Achsen.
    from sklearn.decomposition import PCA

    coords = PCA(n_components=3, random_state=0).fit_transform(embeddings)

    gate = validation.run_gate(embeddings, coords, labels, k=10, min_knn_overlap=0.05)
    assert gate.passed, gate.reasons
    assert gate.label_purity_3d > 0.9


def test_gate_fails_on_random_coordinates():
    """Zufaellige Koordinaten muessen durchfallen, egal wie gut die Embeddings sind."""
    embeddings, labels, _ = synthetic.make_embeddings(
        400, dim=32, n_clusters=4, seed=9, spread=0.25, l2_normalize=False
    )
    coords = np.random.default_rng(3).normal(size=(400, 3))

    gate = validation.run_gate(embeddings, coords, labels, k=10)
    assert not gate.passed
    assert any("Zufallsniveau" in reason for reason in gate.reasons)


def test_gate_fails_when_labels_do_not_separate_in_high_dim():
    """Der wichtigste Fehlerfall: das Problem liegt am Korpus, nicht an UMAP.

    Die Meldung muss das sagen — sonst wird tagelang an n_neighbors gedreht.
    """
    rng = np.random.default_rng(4)
    embeddings = rng.normal(size=(300, 16)).astype(np.float32)
    labels = rng.integers(0, 4, size=300)
    coords = rng.normal(size=(300, 3))

    gate = validation.run_gate(embeddings, coords, labels, k=10)
    assert not gate.passed
    assert any("Korpus" in reason for reason in gate.reasons)


def test_gate_flags_structure_no_better_than_shuffled():
    """Der Fall, den ein Sichtcheck nie faengt.

    Saubere Cluster, die nichts mit dem Label zu tun haben: der Plot sieht
    hervorragend aus, die Zahl sagt Zufallsniveau.
    """
    rng = np.random.default_rng(5)
    embeddings, cluster_labels, _ = synthetic.make_embeddings(
        400, dim=32, n_clusters=4, seed=6, spread=0.2, l2_normalize=False
    )
    from sklearn.decomposition import PCA

    coords = PCA(n_components=3, random_state=0).fit_transform(embeddings)
    # Labels ohne jeden Bezug zu den Clustern.
    unrelated = rng.integers(0, 4, size=400)

    gate = validation.run_gate(
        embeddings,
        coords,
        unrelated,
        k=10,
        shuffled_coords=rng.normal(size=(400, 3)),
    )
    assert not gate.passed
    assert gate.label_purity_3d < 0.4


def test_trustworthiness_subsamples_large_inputs():
    """Ohne Kappung ist das kein langsamer Test, sondern ein sofortiger OOM."""
    rng = np.random.default_rng(6)
    embeddings = rng.normal(size=(2000, 16)).astype(np.float32)
    coords = rng.normal(size=(2000, 3))
    value = validation.trustworthiness(embeddings, coords, k=5, cap=200)
    assert value is not None and 0.0 <= value <= 1.0


def test_gate_summary_and_dict_roundtrip():
    embeddings, labels, _ = synthetic.make_embeddings(
        200, dim=16, n_clusters=4, seed=8, spread=0.25, l2_normalize=False
    )
    from sklearn.decomposition import PCA

    coords = PCA(n_components=3, random_state=0).fit_transform(embeddings)
    gate = validation.run_gate(embeddings, coords, labels, k=5)

    assert "kNN-Overlap" in gate.summary()
    data = gate.to_dict()
    assert data["passed"] is gate.passed
    assert set(data) >= {"knn_overlap", "label_purity_3d", "label_purity_chance", "reasons"}


def test_plot_writes_file(tmp_path):
    embeddings, labels, names = synthetic.make_embeddings(
        150, dim=16, n_clusters=4, seed=10, spread=0.25, l2_normalize=False
    )
    from sklearn.decomposition import PCA

    coords = PCA(n_components=3, random_state=0).fit_transform(embeddings)
    path = validation.plot_gate(
        coords, labels, names, tmp_path / "gate.png",
        shuffled_coords=coords[::-1], title="Test",
    )
    assert path.is_file() and path.stat().st_size > 5000
