"""Kommandozeile der Pipeline.

Aufruf aus dem Repo-Wurzelverzeichnis::

    uv run --project pipeline lse doctor
    uv run --project pipeline lse synth --n 1000 --out data/sample
    uv run --project pipeline lse verify data/sample
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config, ConfigError, load


def _print_section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def _ok(label: str, value: object) -> None:
    print(f"  \033[32m+\033[0m {label:28s} {value}")


def _bad(label: str, value: object) -> None:
    print(f"  \033[31m-\033[0m {label:28s} {value}")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _probe(url: str, timeout: float = 15.0) -> tuple[bool, str]:
    """Prueft einen Host durch den Agent-Proxy hindurch.

    Ein 403 kommt hier von der Egress-Richtlinie, nicht vom Zielserver — der
    Unterschied ist wichtig genug, um ihn im Klartext auszugeben, statt ihn als
    Netzwerkfehler zu tarnen.
    """
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        # Der Host antwortet — 404 oder 405 heisst erreichbar.
        return True, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        reason = str(exc.reason)
        if "403" in reason:
            return False, "durch Egress-Richtlinie gesperrt (403 am Tunnel)"
        return False, reason
    except Exception as exc:  # pragma: no cover - defensiv
        return False, f"{type(exc).__name__}: {exc}"


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"\033[1mLatent Space Explorer\033[0m — Pipeline {__version__}")

    _print_section("Umgebung")
    _ok("Python", sys.version.split()[0])
    for name in ("numpy", "sklearn", "pyarrow", "umap", "torch"):
        try:
            module = __import__(name)
        except ImportError:
            _bad(name, "nicht installiert")
            continue
        _ok(name, getattr(module, "__version__", "?"))

    _print_section("Konfiguration")
    try:
        config = load()
    except ConfigError as exc:
        _bad("config.toml", exc)
        return 1
    _ok("Repo-Wurzel", config.root)
    _ok("Korpus", f"{config.corpus['id']} (Ziel n={config.corpus['target_n']:,})")
    _ok("Modell", f"{config.model['id']} @ {config.model['revision']}")
    _ok(
        "UMAP",
        f"n_neighbors={config.projection['umap_n_neighbors']} "
        f"min_dist={config.projection['umap_min_dist']} "
        f"densmap={config.projection['umap_densmap']}",
    )
    _ok("Label-Feld", config.corpus["label_field"])

    _print_section("Artefakte")
    for label, path in (("Laeufe", config.runs_dir), ("Sample", config.sample_dir)):
        if path.is_dir():
            entries = sorted(p.name for p in path.iterdir())
            _ok(label, f"{path.relative_to(config.root)} ({len(entries)} Eintraege)")
        else:
            _bad(label, f"{path.relative_to(config.root)} fehlt noch")

    if args.network:
        _print_section("Erreichbarkeit (Egress-Richtlinie)")
        targets = {
            "Met-Metadaten": config.corpus["metadata_url"],
            "Met-Objekt-API": config.corpus["object_api"] + "/45734",
            "Met-Bilder": "https://images.metmuseum.org/",
            "Hugging Face": "https://huggingface.co/",
            "PyPI": "https://pypi.org/",
        }
        for label, url in targets.items():
            reachable, detail = _probe(url)
            (_ok if reachable else _bad)(label, detail)
        if os.environ.get("HTTPS_PROXY"):
            print("\n  Hinweis: gesperrte Hosts sind eine Richtlinie dieser Session,")
            print("  kein Projektfehler. Lokal auf deiner Maschine sind sie erreichbar.")

    print()
    return 0


# ---------------------------------------------------------------------------
# synth
# ---------------------------------------------------------------------------


def cmd_synth(args: argparse.Namespace) -> int:
    from . import synthetic
    from .artifacts import ArtifactWriter

    config = load()
    out = Path(args.out)
    if not out.is_absolute():
        out = config.root / out

    n = args.n
    k = int(config.artifacts["neighbors_k"])
    print(f"Erzeuge {n:,} synthetische Punkte ({args.mode}) nach {out} ...")

    if args.mode == "umap":
        embeddings, labels, names = synthetic.make_embeddings(
            n, dim=int(config.model["dim"]), n_clusters=args.clusters, seed=args.seed
        )
        coords = _project(embeddings, config)
        neighbors = synthetic.high_dim_neighbors(embeddings, k)
    else:
        coords, labels, names = synthetic.make_coords(
            n, n_clusters=args.clusters, seed=args.seed
        )
        embeddings = None
        # Ohne hochdimensionalen Raum sind "echte" Nachbarn nicht definiert;
        # der Platzhalter haelt nur das Format offen.
        neighbors = np.tile(np.arange(min(k, n), dtype=np.uint32), (n, 1))

    writer = ArtifactWriter(out, config)
    spec = writer.add_coords(coords)
    writer.add_labels(labels, names)
    writer.add_colors(synthetic.palette_colors(labels))
    writer.add_neighbors(neighbors)
    if embeddings is not None and args.keep_embeddings:
        writer.add_embeddings(embeddings)

    manifest = writer.finish(
        corpus={
            "id": "synthetic",
            "note": (
                "Synthetische Gauss-Blobs. Kein echter Korpus — dient dazu, den "
                "Viewer unabhaengig von Modell und Bilddaten zu verifizieren."
            ),
            "n_clusters": args.clusters,
            "seed": args.seed,
        },
        model={"id": "none", "revision": "none", "dim": int(config.model["dim"])},
        projection=(
            dict(config.projection) if args.mode == "umap" else {"method": "direct-3d-blobs"}
        ),
    )

    print(f"  n           {manifest.n:,}")
    print(f"  Radius      {spec.radius:.2f} (Box +/-{spec.box:g})")
    print(f"  Dateien     {len(manifest.files)}")
    total = sum(f["bytes"] for f in manifest.files)
    print(f"  Groesse     {total / 1024:.1f} KiB")
    return 0


def _project(embeddings: np.ndarray, config: Config) -> np.ndarray:
    """L2 -> PCA -> UMAP, in genau dieser Reihenfolge.

    PCA vorschalten ist nicht primaer Qualitaet, sondern Machbarkeit: es macht
    den kNN-Bau um eine Groessenordnung schneller und das persistierte Modell um
    eine Groessenordnung kleiner.
    """
    import umap
    from sklearn.decomposition import PCA

    proj = config.projection
    x = np.asarray(embeddings, dtype=np.float32)

    if proj["l2_normalize"]:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)

    components = min(int(proj["pca_components"]), x.shape[1], x.shape[0])
    pca = PCA(n_components=components, whiten=False, random_state=0)
    x = pca.fit_transform(x)
    print(f"  PCA {components}d, erklaerte Varianz {pca.explained_variance_ratio_.sum():.3f}")

    seed = proj["random_state"] or None
    reducer = umap.UMAP(
        n_components=int(proj["umap_n_components"]),
        n_neighbors=int(proj["umap_n_neighbors"]),
        min_dist=float(proj["umap_min_dist"]),
        metric=str(proj["umap_metric"]),
        init=str(proj["umap_init"]),
        n_epochs=int(proj["umap_n_epochs"]),
        densmap=bool(proj["umap_densmap"]),
        random_state=seed,
        # Unter n=4096 schaltet UMAP auf _small_data und laesst den kNN-Index
        # None — transform() wuerde dann werfen. Erzwingen, damit der Pfad auch
        # bei kleinen Testlaeufen derselbe ist.
        force_approximation_algorithm=True,
        verbose=False,
    )
    return np.asarray(reducer.fit_transform(x), dtype=np.float64)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    from .artifacts import ArtifactError, load_manifest, verify

    config = load()
    target = Path(args.path)
    if not target.is_absolute():
        target = config.root / target

    try:
        manifest = load_manifest(target)
        problems = verify(target)
    except ArtifactError as exc:
        _bad("Artefaktsatz", exc)
        return 1

    print(f"\033[1m{target}\033[0m")
    _ok("n", f"{manifest.n:,}")
    _ok("erzeugt", manifest.created_at)
    _ok("Korpus", manifest.corpus.get("id", "?"))
    _ok("Commit", (manifest.provenance.get("git_commit") or "—")[:12])

    if problems:
        print()
        for problem in problems:
            _bad("Problem", problem)
        return 1

    _ok("Pruefsummen", f"{len(manifest.files)} Dateien in Ordnung")
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lse", description=__doc__)
    parser.add_argument("--version", action="version", version=f"lse {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Umgebung, Konfiguration und Erreichbarkeit pruefen")
    doctor.add_argument(
        "--network",
        action="store_true",
        help="zusaetzlich pruefen, welche Hosts die Egress-Richtlinie zulaesst",
    )
    doctor.set_defaults(func=cmd_doctor)

    synth = sub.add_parser("synth", help="synthetischen Artefaktsatz erzeugen")
    synth.add_argument("--n", type=int, default=1000, help="Anzahl Punkte (Vorgabe 1000)")
    synth.add_argument("--clusters", type=int, default=8, help="Anzahl Cluster (Vorgabe 8)")
    synth.add_argument("--seed", type=int, default=0)
    synth.add_argument(
        "--mode",
        choices=("coords", "umap"),
        default="coords",
        help="coords: direkte 3D-Blobs (schnell). umap: echter L2->PCA->UMAP-Pfad.",
    )
    synth.add_argument(
        "--keep-embeddings",
        action="store_true",
        help="hochdimensionale Embeddings mitschreiben (nur bei --mode umap)",
    )
    synth.add_argument("--out", default="data/sample", help="Zielverzeichnis")
    synth.set_defaults(func=cmd_synth)

    verify_cmd = sub.add_parser("verify", help="Artefaktsatz gegen sein Manifest pruefen")
    verify_cmd.add_argument("path", help="Verzeichnis mit manifest.json")
    verify_cmd.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"\033[31mKonfigurationsfehler:\033[0m {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
