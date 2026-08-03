"""Synthetische Daten in exakt den Zielformaten.

Kein Spielzeug, sondern Infrastruktur. Zwei Gruende:

1. Es entkoppelt den Viewer vollstaendig von Korpus und Modell. Der Renderer
   kann bei 1 Mio. Punkten belastet werden, bevor ein einziges echtes Embedding
   existiert — und Phase 3 muss nicht auf Phase 1 warten.
2. In der Entwicklungs-Session sind ``huggingface.co`` und die Met-Bild-Hosts
   durch die Egress-Richtlinie gesperrt (siehe docs/decisions.md E8). Ohne
   synthetische Daten waere der Viewer dort nicht verifizierbar, nur schreibbar.

Die erzeugten Blobs sind bewusst *zu sauber*: gut getrennte Gaussverteilungen
mit bekannten Labels. Das macht sie zum Testfall fuer den Renderer und
gleichzeitig zur Warnung — genau so sieht auch reines Rauschen nach UMAP aus,
weshalb der Shuffle-Control in Phase 0b noetig ist.
"""

from __future__ import annotations

import numpy as np

# Okabe-Ito, die etablierte farbenblind-sichere kategoriale Palette, ohne
# Schwarz. Bei 3px loest das Auge Saettigung auf, nicht Farbton — deshalb
# saettigungsstarke Werte, keine Pastelltoene.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (230, 159, 0),    # Orange
    (86, 180, 233),   # Himmelblau
    (0, 158, 115),    # Blaugruen
    (240, 228, 66),   # Gelb
    (0, 114, 178),    # Blau
    (213, 94, 0),     # Zinnoberrot
    (204, 121, 167),  # Rotviolett
    (148, 103, 189),  # Violett (Ergaenzung)
)


def palette_colors(labels: np.ndarray) -> np.ndarray:
    """Bildet Labelindizes auf RGB ab.

    Ueber ``len(PALETTE)`` Klassen hinaus wiederholen sich die Farben. Das ist
    Absicht: mehr als acht kategoriale Farben kann niemand auseinanderhalten.
    Bei mehr Klassen gehoert eine Gruppierung in den Korpusschritt, nicht eine
    groessere Palette hierher.
    """
    labels = np.asarray(labels, dtype=np.int64)
    lut = np.array(PALETTE, dtype=np.uint8)
    return lut[labels % len(lut)]


def _noise_sigma(spread: float, dim: int) -> float:
    """Streuung pro Dimension, so dass der Rauschvektor die Norm ``spread`` hat.

    Ohne diese Umrechnung waere ``spread`` dimensionsabhaengig und in hohen
    Dimensionen katastrophal: bei ``sigma=0.35`` pro Dimension und ``dim=768``
    hat der Rauschvektor die Norm ``0.35 * sqrt(768) ~ 9.7``, waehrend zwei
    zufaellige Einheitsvektoren nur ``~1.41`` auseinanderliegen. Die "Cluster"
    waeren dann reines Rauschen — und der Generator wuerde genau die Illusion
    erzeugen, gegen die das Go/No-Go-Tor in Phase 0b schuetzt.

    So interpretiert ist ``spread`` die Rauschnorm relativ zum
    Einheits-Zentrumsabstand und bedeutet in jeder Dimension dasselbe.
    """
    return spread / np.sqrt(dim)


def make_embeddings(
    n: int,
    dim: int = 768,
    n_clusters: int = 8,
    *,
    seed: int = 0,
    spread: float = 0.35,
    l2_normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Hochdimensionale Blobs mit bekannten Labels.

    Laeuft durch denselben PCA/UMAP-Pfad wie echte Embeddings, testet also den
    echten Code und nicht eine Abkuerzung.

    ``spread`` ist die **Norm** des Rauschvektors, nicht die Streuung pro
    Dimension — siehe :func:`_noise_sigma`. Zwei zufaellige Zentren liegen etwa
    1,41 auseinander; ``spread`` deutlich darunter heisst getrennte Cluster.

    Rueckgabe: ``(embeddings [n, dim] float32, labels [n] int64, label_names)``.
    """
    if n <= 0:
        raise ValueError("n muss positiv sein.")
    if n_clusters <= 0:
        raise ValueError("n_clusters muss positiv sein.")

    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    labels = rng.integers(0, n_clusters, size=n)
    sigma = _noise_sigma(spread, dim)
    embeddings = centers[labels] + rng.normal(scale=sigma, size=(n, dim)).astype(np.float32)

    if l2_normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        np.divide(embeddings, norms, out=embeddings, where=norms > 0)

    names = [f"cluster-{i:02d}" for i in range(n_clusters)]
    return embeddings.astype(np.float32), labels, names


def make_coords(
    n: int,
    n_clusters: int = 8,
    *,
    seed: int = 0,
    spread: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fertige 3D-Blobs, ohne den Umweg ueber UMAP.

    Fuer Renderer-Lasttests: bei 1 Mio. Punkten dauert echtes UMAP Stunden, der
    Renderer soll aber jetzt gemessen werden. Die Punktverteilung ist einer
    UMAP-Ausgabe aehnlich genug, um Fuellrate und Picking realistisch zu belasten.

    ``spread`` bedeutet dasselbe wie in :func:`make_embeddings`: die Norm des
    Rauschvektors relativ zum Einheits-Zentrumsabstand.
    """
    if n <= 0:
        raise ValueError("n muss positiv sein.")

    rng = np.random.default_rng(seed)
    # Cluster auf einer Kugelschale statt im Wuerfel: naeher an dem, was UMAP
    # liefert, und es erzeugt die dichten Regionen, die den Fuellratentest
    # ueberhaupt aussagekraeftig machen.
    centers = rng.normal(size=(n_clusters, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    labels = rng.integers(0, n_clusters, size=n)
    sigma = _noise_sigma(spread, 3)
    coords = centers[labels] + rng.normal(scale=sigma, size=(n, 3))

    names = [f"cluster-{i:02d}" for i in range(n_clusters)]
    return coords.astype(np.float64), labels, names


def make_titles(labels: np.ndarray, names: list[str]) -> list[str]:
    """Platzhaltertitel, damit die Stichwortsuche etwas zu finden hat."""
    return [f"{names[int(label)]} #{i:06d}" for i, label in enumerate(labels)]


def high_dim_neighbors(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Top-k naechste Nachbarn im hochdimensionalen Raum, per Cosine.

    Brute Force. Fuer die synthetischen Groessen ausreichend; der echte Lauf
    nimmt den kNN-Graphen, den UMAP ohnehin baut.
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    n = len(embeddings)
    k = min(k, max(n - 1, 1))

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)

    out = np.empty((n, k), dtype=np.uint32)
    # Blockweise, damit die Aehnlichkeitsmatrix nicht n^2 Speicher braucht.
    block = max(1, min(n, 4096))
    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = normalized[start:stop] @ normalized.T
        # Sich selbst nicht als eigenen Nachbarn zurueckgeben.
        for row, idx in enumerate(range(start, stop)):
            sims[row, idx] = -np.inf
        top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        order = np.take_along_axis(sims, top, axis=1).argsort(axis=1)[:, ::-1]
        out[start:stop] = np.take_along_axis(top, order, axis=1).astype(np.uint32)

    return out
