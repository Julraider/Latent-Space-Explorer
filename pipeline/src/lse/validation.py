"""Das Go/No-Go-Tor: bedeutet dieser Raum ueberhaupt etwas?

Der ursprüngliche Plan sah als Kontrolle einen Blick auf einen Plot vor. Das
reicht nicht, und zwar aus einem konkreten Grund: **UMAP erzeugt aus reinem
Rauschen ueberzeugend aussehende Cluster.** Ein Sichtcheck kann also nicht
durchfallen und ist als Kontrolle wertlos.

Geprueft wird deshalb dreifach:

1. **Shuffle-Control** — dieselbe Pipeline auf strukturlos gemachten Daten. Was
   dort an Struktur erscheint, ist die Artefakt-Baseline des Verfahrens.
2. **kNN-Overlap** — wie viele der echten hochdimensionalen Nachbarn eines
   Punktes sind auch in 3D seine Nachbarn.
3. **Label-Reinheit** — teilen die 3D-Nachbarn das bekannte Label.

Wenn bekannte Labels sich nicht trennen, ist der **Korpus** falsch, nicht die
Parameter. Das ist in dieser Phase eine Entscheidung von einem Nachmittag; nach
Phase 3 kostet sie einen Renderer-Neuschrieb.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def shuffle_columns(embeddings: np.ndarray, seed: int = 0) -> np.ndarray:
    """Zerstoert die Struktur, erhaelt die Randverteilungen.

    **Zeilen zu mischen waere wirkungslos** — es ist dieselbe Punktmenge in
    anderer Reihenfolge, und UMAP faende exakt dieselbe Mannigfaltigkeit. Die
    Kontrolle muss die Korrelationen *zwischen* den Dimensionen aufloesen, und
    genau das macht eine unabhaengige Permutation je Spalte: jede Dimension
    behaelt ihre Verteilung, aber kein Punkt ist mehr eine sinnvolle Kombination.
    """
    rng = np.random.default_rng(seed)
    shuffled = np.array(embeddings, dtype=np.float32, copy=True)
    for column in range(shuffled.shape[1]):
        rng.shuffle(shuffled[:, column])
    return shuffled


def knn_indices_3d(coords: np.ndarray, k: int) -> np.ndarray:
    """Top-k Nachbarn in 3D, ohne den Punkt selbst."""
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    tree = cKDTree(coords)
    # k+1, weil der naechste Nachbar der Punkt selbst ist.
    _, indices = tree.query(coords, k=min(k + 1, len(coords)))
    if indices.ndim == 1:
        indices = indices[:, None]
    return indices[:, 1:]


def knn_indices_high(embeddings: np.ndarray, k: int, *, block: int = 2048) -> np.ndarray:
    """Top-k Nachbarn im hochdimensionalen Raum, per Cosine.

    Blockweise, damit die Aehnlichkeitsmatrix nicht n^2 Speicher braucht.
    """
    x = np.ascontiguousarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)

    n = len(x)
    k = min(k, max(n - 1, 1))
    out = np.empty((n, k), dtype=np.int64)

    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = x[start:stop] @ x.T
        rows = np.arange(stop - start)
        sims[rows, np.arange(start, stop)] = -np.inf
        top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        order = np.take_along_axis(sims, top, axis=1).argsort(axis=1)[:, ::-1]
        out[start:stop] = np.take_along_axis(top, order, axis=1)

    return out


def knn_overlap(high: np.ndarray, low: np.ndarray) -> float:
    """Mittlerer Anteil gemeinsamer Nachbarn zwischen hochdimensional und 3D.

    **Zur Einordnung: von 768d auf 3d sind 0,10 bis 0,25 normal und in
    Ordnung.** Mehr ist geometrisch nicht drin — drei Dimensionen koennen die
    Nachbarschaften eines 768-dimensionalen Raums nicht tragen. Die Zahl dient
    dem Vergleich von Konfigurationen und dem Erkennen von Katastrophen; ein
    Wert um 0,02 heisst Rauschen.
    """
    k = min(high.shape[1], low.shape[1])
    shared = [
        len(set(high[i, :k].tolist()) & set(low[i, :k].tolist())) for i in range(len(high))
    ]
    return float(np.mean(shared) / k)


def label_purity(neighbors: np.ndarray, labels: np.ndarray) -> float:
    """Anteil der Nachbarn, die das Label des Punktes teilen."""
    labels = np.asarray(labels)
    return float((labels[neighbors] == labels[:, None]).mean())


def trustworthiness(embeddings: np.ndarray, coords: np.ndarray, k: int, *, cap: int = 5000,
                    seed: int = 0) -> float | None:
    """sklearn-Trustworthiness auf einer Unterstichprobe.

    Die Funktion berechnet **alle paarweisen Abstaende** — O(n^2) Speicher. Bei
    500k Punkten waeren das rund 2 TB. Ohne die Kappung ist das kein langsamer
    Test, sondern ein sofortiger OOM.
    """
    from sklearn.manifold import trustworthiness as sk_trustworthiness

    n = len(embeddings)
    if n <= k + 1:
        return None
    if n > cap:
        rng = np.random.default_rng(seed)
        pick = rng.choice(n, size=cap, replace=False)
        embeddings, coords = embeddings[pick], coords[pick]
    return float(sk_trustworthiness(embeddings, coords, n_neighbors=k))


@dataclass
class GateResult:
    """Das Ergebnis des Go/No-Go-Tors."""

    k: int
    n: int
    knn_overlap: float
    knn_overlap_shuffled: float | None
    label_purity_high: float
    label_purity_3d: float
    label_purity_shuffled: float | None
    label_purity_chance: float
    trustworthiness: float | None
    passed: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"  n                        {self.n:,}   k = {self.k}",
            f"  kNN-Overlap hoch -> 3D   {self.knn_overlap:.3f}"
            + (
                f"   (gemischt: {self.knn_overlap_shuffled:.3f})"
                if self.knn_overlap_shuffled is not None
                else ""
            ),
            f"  Label-Reinheit hoch      {self.label_purity_high:.3f}",
            f"  Label-Reinheit 3D        {self.label_purity_3d:.3f}"
            + (
                f"   (gemischt: {self.label_purity_shuffled:.3f})"
                if self.label_purity_shuffled is not None
                else ""
            ),
            f"  Zufallsniveau            {self.label_purity_chance:.3f}",
        ]
        if self.trustworthiness is not None:
            lines.append(f"  Trustworthiness          {self.trustworthiness:.3f}")
        return "\n".join(lines)


def run_gate(
    embeddings: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 15,
    shuffled_coords: np.ndarray | None = None,
    shuffled_embeddings: np.ndarray | None = None,
    min_knn_overlap: float = 0.05,
    high_knn: np.ndarray | None = None,
) -> GateResult:
    """Rechnet alle Kennzahlen und faellt das Urteil."""
    n = len(coords)
    k = min(k, max(n - 1, 1))

    high = high_knn[:, :k] if high_knn is not None else knn_indices_high(embeddings, k)
    low = knn_indices_3d(coords, k)

    overlap = knn_overlap(high, low)
    purity_high = label_purity(high, labels)
    purity_3d = label_purity(low, labels)

    # Zufallsniveau ist nicht 1/Klassenzahl, sondern die Summe der quadrierten
    # Klassenanteile — bei unausgewogenen Klassen ist das deutlich hoeher.
    _, counts = np.unique(labels, return_counts=True)
    chance = float(((counts / counts.sum()) ** 2).sum())

    overlap_shuffled = None
    purity_shuffled = None
    if shuffled_coords is not None:
        low_shuffled = knn_indices_3d(shuffled_coords, k)
        purity_shuffled = label_purity(low_shuffled, labels)
        if shuffled_embeddings is not None:
            high_shuffled = knn_indices_high(shuffled_embeddings, k)
            overlap_shuffled = knn_overlap(high_shuffled, low_shuffled)

    reasons: list[str] = []
    if overlap < min_knn_overlap:
        reasons.append(
            f"kNN-Overlap {overlap:.3f} unter der Untergrenze {min_knn_overlap:.3f} — "
            "die Projektion bildet die Nachbarschaften nicht ab."
        )
    if purity_3d <= chance + 0.05:
        reasons.append(
            f"Label-Reinheit in 3D ({purity_3d:.3f}) kaum ueber dem Zufallsniveau "
            f"({chance:.3f}) — die Cluster tragen keine Bedeutung."
        )
    if purity_high <= chance + 0.05:
        reasons.append(
            f"Schon im hochdimensionalen Raum trennen die Labels nicht "
            f"({purity_high:.3f} gegen {chance:.3f} Zufall). Das ist ein Problem "
            "des Korpus oder des Modells, nicht der Projektion — Parameter zu "
            "tunen hilft hier nicht."
        )
    if purity_shuffled is not None and purity_3d <= purity_shuffled + 0.05:
        reasons.append(
            f"Die Struktur ist nicht besser als bei strukturlos gemachten Daten "
            f"({purity_3d:.3f} gegen {purity_shuffled:.3f}) — was zu sehen ist, "
            "ist ein Artefakt des Verfahrens."
        )

    return GateResult(
        k=k,
        n=n,
        knn_overlap=overlap,
        knn_overlap_shuffled=overlap_shuffled,
        label_purity_high=purity_high,
        label_purity_3d=purity_3d,
        label_purity_shuffled=purity_shuffled,
        label_purity_chance=chance,
        trustworthiness=trustworthiness(embeddings, coords, k),
        passed=not reasons,
        reasons=reasons,
    )


def plot_gate(
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    path: Path,
    *,
    shuffled_coords: np.ndarray | None = None,
    title: str = "",
) -> Path:
    """Nebeneinander: echte Projektion und Shuffle-Control.

    Bewusst in 2D (zwei der drei Achsen), nicht als 3D-Scatter: matplotlibs
    3D-Tiefensortierung arbeitet nach dem Maleralgorithmus und ist damit
    schlicht falsch — getrennte Cluster verschmelzen optisch, und der Plot
    wuerde eine Trennung verschweigen, die es gibt.

    Der Vergleich ist der Punkt. Allein betrachtet sieht die rechte Seite
    genauso ueberzeugend aus wie die linke.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .synthetic import PALETTE

    panels = 2 if shuffled_coords is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(7 * panels, 6.5), squeeze=False)
    colors = np.array(PALETTE, dtype=float) / 255.0

    datasets = [(coords, "Echte Embeddings")]
    if shuffled_coords is not None:
        datasets.append((shuffled_coords, "Shuffle-Control (strukturlos)"))

    for axis, (data, name) in zip(axes[0], datasets):
        for index in np.unique(labels):
            mask = labels == index
            axis.scatter(
                data[mask, 0],
                data[mask, 1],
                s=6,
                alpha=0.75,
                linewidths=0,
                color=colors[index % len(colors)],
                label=label_names[index] if index < len(label_names) else str(index),
            )
        axis.set_title(name)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal", adjustable="datalim")

    axes[0][0].legend(fontsize=7, markerscale=2, loc="best", framealpha=0.85)
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
