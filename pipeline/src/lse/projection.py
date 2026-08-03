"""Von Embeddings zu 3D-Koordinaten: L2 -> PCA -> UMAP.

Die Reihenfolge ist bindend und in :mod:`lse.config` durchgesetzt.

**PCA vorschalten ist nicht primaer eine Qualitaetsfrage, sondern eine
Machbarkeitsfrage.** Drei Auszahlungen:

* kNN-Bau um eine Groessenordnung schneller, und erst dadurch passen 500k
  Punkte ueberhaupt in den Arbeitsspeicher (ohne PCA sind 500k x 1152 float32
  allein 2,3 GB Eingabe, plus pynndescents RP-Forest).
* Das persistierte Modell schrumpft von hunderten MB auf zweistellige — das
  ist der Unterschied zwischen "Live-Insert ausliefer.bar" und "nicht".
* Leichtes Denoising, das die sichtbare Clusterstruktur meist verbessert.

Nicht whitenen und danach nicht erneut normalisieren: beides skaliert die Achsen
um und zerstoert genau die Nachbarschaftsstruktur, die abgebildet werden soll.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PCA_FILE = "pca.npz"


@dataclass
class ProjectionResult:
    """Koordinaten plus alles, was zur Reproduktion und zum Live-Insert noetig ist."""

    coords: np.ndarray
    pca_components: int
    explained_variance: float
    seconds: float
    params: dict[str, Any] = field(default_factory=dict)
    # Der kNN-Graph, den UMAP ohnehin baut — im HOCHDIMENSIONALEN Raum.
    # Traegt die Ehrlichkeitsschicht und spart einen zweiten kNN-Lauf.
    knn_indices: np.ndarray | None = None


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Zeilenweise auf Einheitslaenge; Nullzeilen bleiben null."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def fit_pca(x: np.ndarray, n_components: int, seed: int = 0):
    """PCA anpassen; gibt das Modell und die transformierten Daten zurueck."""
    from sklearn.decomposition import PCA

    components = min(n_components, x.shape[1], x.shape[0])
    pca = PCA(n_components=components, whiten=False, random_state=seed)
    return pca, pca.fit_transform(x)


def save_pca(pca, path: Path) -> Path:
    """Persistiert den PCA-Zustand — ohne Pickle.

    Der Live-Insert braucht spaeter genau diese Transformation, um einen neuen
    Prompt in denselben Raum zu bringen. Als Pickle waere sie ueber
    numpy-Versionen hinweg fragil; drei Arrays sind es nicht.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )
    return path


def load_pca(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as data:
        return {key: data[key] for key in data.files}


def apply_pca(x: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    """Wendet einen gespeicherten PCA-Zustand an — dieselbe Rechnung wie sklearn."""
    return (np.asarray(x, dtype=np.float32) - state["mean"]) @ state["components"].T


def project(
    embeddings: np.ndarray,
    params: dict[str, Any],
    *,
    keep_knn: bool = True,
    verbose: bool = False,
) -> tuple[ProjectionResult, Any, Any]:
    """Fuehrt L2 -> PCA -> UMAP aus.

    Rueckgabe: Ergebnis, das PCA-Modell und der UMAP-Reducer. Die beiden Modelle
    werden vom Aufrufer persistiert; hier bleibt nichts liegen.
    """
    import umap

    started = time.perf_counter()
    x = np.asarray(embeddings, dtype=np.float32)

    if params.get("l2_normalize", True):
        x = l2_normalize(x)

    pca, x = fit_pca(x, int(params["pca_components"]))
    explained = float(pca.explained_variance_ratio_.sum())
    if verbose:
        print(f"  PCA {x.shape[1]}d, erklaerte Varianz {explained:.3f}")
        if explained < 0.6:
            print("  Hinweis: unter 0,60 — mehr Komponenten waeren angebracht.")

    # random_state erzwingt n_jobs=1 quer durch kNN-Bau und Layout (umap_.py:2003),
    # grob Faktor 3-6 Wall-Clock. Null bedeutet hier: unseeded und parallel,
    # also der Explorationsmodus.
    seed = params.get("random_state") or None

    reducer = umap.UMAP(
        n_components=int(params["umap_n_components"]),
        n_neighbors=int(params["umap_n_neighbors"]),
        min_dist=float(params["umap_min_dist"]),
        metric=str(params["umap_metric"]),
        init=str(params["umap_init"]),
        n_epochs=int(params["umap_n_epochs"]),
        densmap=bool(params.get("umap_densmap", False)),
        random_state=seed,
        # Unter n=4096 schaltet UMAP auf _small_data und laesst
        # `_knn_search_index` auf None — `transform()` wuerde dann werfen. Der
        # Pfad muss bei kleinen Testlaeufen derselbe sein wie spaeter, sonst
        # faellt genau das erst in Phase 4 auf.
        force_approximation_algorithm=True,
        verbose=verbose,
    )
    coords = np.asarray(reducer.fit_transform(x), dtype=np.float64)

    knn = None
    if keep_knn:
        # UMAP hat den kNN-Graphen im hochdimensionalen Raum bereits gebaut.
        # Ihn hier abzugreifen spart einen kompletten zweiten Durchlauf.
        knn = getattr(reducer, "_knn_indices", None)
        if knn is not None:
            knn = np.asarray(knn)

    return (
        ProjectionResult(
            coords=coords,
            pca_components=int(x.shape[1]),
            explained_variance=explained,
            seconds=time.perf_counter() - started,
            params=dict(params),
            knn_indices=knn,
        ),
        pca,
        reducer,
    )
