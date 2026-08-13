"""Wo landet ein neuer Prompt im Raum?

Die naheliegende Antwort — den Prompt-Vektor durch ``umap.transform()`` schicken
— ist hier die falsche. Drei Gruende, jeder fuer sich ausreichend
(docs/decisions.md E4):

* **Modality Gap.** CLIP-artige Modelle legen Bild- und Text-Embeddings in
  *verschiedene Kegel* des gemeinsamen Raums. Ein transformierter Text-Prompt
  landet dadurch systematisch versetzt, moeglicherweise ausserhalb der Wolke.
  Bei einem Bildkorpus mit Textsuche ist das kein Randfall, sondern der
  Normalfall.
* **Modellgroesse.** ``transform()`` braucht ``_raw_data`` und den
  pynndescent-Index; ein Pickle schleppt die komplette Trainingsmatrix mit.
* **Versionsfragilitaet.** UMAP-Pickles brechen ueber numpy-/numba-Wechsel.

Stattdessen: die naechsten **Bild**-Nachbarn des Prompts im hochdimensionalen
Raum suchen und ihn auf deren Schwerpunkt in 3D setzen. Das umgeht alle drei
Probleme und ist in der Oberflaeche ehrlicher erklaerbar ("platziert zwischen
seinen naechsten Nachbarn") als eine undurchsichtige Projektion.

Zum Modality Gap noch einmal deutlich: er verschiebt die *absoluten*
Cosine-Werte zwischen Text und Bild nach unten — eine Text-Bild-Aehnlichkeit von
0,25 kann exzellent sein, waehrend 0,25 zwischen zwei Bildern schlecht waere.
Die *Rangfolge* bleibt aber aussagekraeftig, und nur die wird hier benutzt.
Absolute Schwellwerte auf diese Zahlen anzuwenden waere ein Fehler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Placement:
    """Ergebnis einer Prompt-Platzierung."""

    position: list[float]
    neighbors: list[int]
    similarities: list[float]
    weights: list[float]
    # Streuung der Nachbarn in 3D, gemessen am Wolkenradius. Hoch heisst: die
    # naechsten Nachbarn liegen im Raum weit auseinander, die Platzierung ist
    # also ein Kompromiss zwischen entfernten Regionen — und der Schwerpunkt
    # liegt womoeglich in einer Leerstelle, in der gar nichts ist.
    spread: float
    confident: bool
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        norm = float(np.linalg.norm(x))
        return x / norm if norm > 0 else x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def top_k_neighbors(
    query: np.ndarray, embeddings: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k nach Cosine. Erwartet bereits normalisierte Eingaben."""
    similarities = embeddings @ query
    k = min(k, len(similarities))
    top = np.argpartition(-similarities, kth=k - 1)[:k]
    order = np.argsort(-similarities[top])
    top = top[order]
    return top, similarities[top]


def softmax_weights(similarities: np.ndarray, temperature: float = 0.05) -> np.ndarray:
    """Gewichte aus Aehnlichkeiten.

    Softmax statt gleicher Gewichte, weil der beste Treffer oft deutlich besser
    ist als der zwanzigste — und ein ungewichteter Schwerpunkt den Prompt dann
    von seinem eigentlichen Ziel wegzieht.

    Die Temperatur wird auf die *Spannweite* der Aehnlichkeiten bezogen, nicht
    auf ihre absoluten Werte: wegen des Modality Gap liegen Text-Bild-Werte in
    einem anderen Bereich als Bild-Bild-Werte, und eine feste Temperatur waere
    in einem der beiden Faelle sinnlos.
    """
    similarities = np.asarray(similarities, dtype=np.float64)
    spread = float(similarities.max() - similarities.min())
    scale = max(spread, 1e-6) * max(temperature, 1e-6) / 0.05 if spread > 0 else 1.0
    logits = (similarities - similarities.max()) / max(scale, 1e-6)
    weights = np.exp(logits)
    total = weights.sum()
    return weights / total if total > 0 else np.full_like(weights, 1.0 / len(weights))


def place(
    query: np.ndarray,
    embeddings: np.ndarray,
    coords: np.ndarray,
    *,
    k: int = 12,
    radius: float | None = None,
    spread_limit: float = 0.55,
    temperature: float = 0.05,
) -> Placement:
    """Platziert einen Anfragevektor im 3D-Raum.

    ``embeddings`` und ``coords`` muessen dieselbe Zeilenreihenfolge haben —
    die ID-Invariante des Projekts.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float64)
    if len(embeddings) != len(coords):
        raise ValueError(
            f"{len(embeddings)} Embeddings gegen {len(coords)} Koordinaten — "
            "die ID-Invariante ist verletzt."
        )
    if len(embeddings) == 0:
        raise ValueError("Leerer Korpus.")

    query = l2_normalize(np.asarray(query, dtype=np.float32).ravel())
    normalized = l2_normalize(embeddings)

    neighbors, similarities = top_k_neighbors(query, normalized, k)
    weights = softmax_weights(similarities, temperature)

    picked = coords[neighbors]
    position = (picked * weights[:, None]).sum(axis=0)

    # Streuung als mittlerer Abstand der Nachbarn vom Schwerpunkt.
    distances = np.linalg.norm(picked - position, axis=1)
    scale = radius if radius and radius > 0 else float(np.linalg.norm(coords, axis=1).max() or 1.0)
    spread = float(distances.mean() / scale)

    confident = spread <= spread_limit
    note = (
        ""
        if confident
        else (
            "Die naechsten Nachbarn liegen im Raum weit auseinander. Der Punkt "
            "sitzt zwischen mehreren Regionen und moeglicherweise in einer "
            "Leerstelle — die Projektion kann diese Anfrage nicht an einer "
            "einzigen Stelle abbilden."
        )
    )

    return Placement(
        position=[float(v) for v in position],
        neighbors=[int(v) for v in neighbors],
        similarities=[float(v) for v in similarities],
        weights=[float(v) for v in weights],
        spread=spread,
        confident=confident,
        note=note,
    )
