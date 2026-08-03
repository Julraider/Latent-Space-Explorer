"""Kommandozeile der Pipeline.

Aufruf aus dem Repo-Wurzelverzeichnis::

    uv run --project pipeline lse doctor --network
    uv run --project pipeline lse fetch   --n 1000        # Metadaten -> Auswahl
    uv run --project pipeline lse images                  # lokal: Bilder holen
    uv run --project pipeline lse embed                   # lokal: SigLIP 2
    uv run --project pipeline lse project                 # Projektion + Tor
    uv run --project pipeline lse verify data/runs/dev

`images` und `embed` brauchen Hosts, die in der Entwicklungs-Session gesperrt
sind (siehe docs/decisions.md E8) — sie laufen auf der lokalen Maschine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config, ConfigError, load

DEFAULT_RUN = "data/runs/dev"


def _section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def _ok(label: str, value: object) -> None:
    print(f"  \033[32m+\033[0m {label:26s} {value}")


def _bad(label: str, value: object) -> None:
    print(f"  \033[31m-\033[0m {label:26s} {value}")


def _resolve(config: Config, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config.root / path


def _bar(done: int, total: int, width: int = 32) -> None:
    if not total:
        return
    filled = int(width * done / total)
    pct = 100 * done / total
    end = "\n" if done >= total else ""
    print(f"\r  [{'#' * filled}{'.' * (width - filled)}] {pct:5.1f}%  {done:,}/{total:,}",
          end=end, flush=True)


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

    _section("Umgebung")
    _ok("Python", sys.version.split()[0])
    for name in ("numpy", "sklearn", "pyarrow", "umap", "matplotlib", "torch", "transformers"):
        try:
            module = __import__(name)
        except ImportError:
            _bad(name, "nicht installiert")
            continue
        _ok(name, getattr(module, "__version__", "?"))

    _section("Konfiguration")
    try:
        config = load()
    except ConfigError as exc:
        _bad("config.toml", exc)
        return 1
    _ok("Repo-Wurzel", config.root)
    _ok("Korpus", f"{config.corpus['id']} (Ziel n={config.corpus['target_n']:,})")
    _ok("Modell", f"{config.model['id']} @ {config.model['revision']}")
    _ok("Label / Facetten", f"{config.corpus['label_field']} / {config.corpus['facets']}")
    _ok(
        "UMAP",
        f"n_neighbors={config.projection['umap_n_neighbors']} "
        f"min_dist={config.projection['umap_min_dist']} "
        f"densmap={config.projection['umap_densmap']}",
    )

    _section("Artefakte")
    for label, path in (("Laeufe", config.runs_dir), ("Sample", config.sample_dir)):
        if path.is_dir():
            _ok(label, f"{path.relative_to(config.root)} ({len(list(path.iterdir()))} Eintraege)")
        else:
            _bad(label, f"{path.relative_to(config.root)} fehlt noch")

    if args.network:
        _section("Erreichbarkeit (Egress-Richtlinie)")
        for label, url in {
            "Met-Metadaten": config.corpus["metadata_url"],
            "Met-Objekt-API": config.corpus["object_api"] + "/45734",
            "Met-Bilder": "https://images.metmuseum.org/",
            "Hugging Face": "https://huggingface.co/",
            "PyPI": "https://pypi.org/",
        }.items():
            reachable, detail = _probe(url)
            (_ok if reachable else _bad)(label, detail)
        if os.environ.get("HTTPS_PROXY"):
            print("\n  Gesperrte Hosts sind eine Richtlinie dieser Session, kein")
            print("  Projektfehler. `images` und `embed` gehoeren auf die lokale Maschine.")

    print()
    return 0


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def cmd_fetch(args: argparse.Namespace) -> int:
    from . import corpus as corpus_module

    config = load()
    run = _resolve(config, args.out)
    raw = config.root / "data" / "raw" / "MetObjects.csv"

    print(f"Metadaten -> {raw.relative_to(config.root)}")
    corpus_module.download_metadata(
        config.corpus["metadata_url"], raw, force=args.force, progress=_bar
    )
    _ok("CSV", f"{raw.stat().st_size / 1e6:.0f} MB")

    print("\nLese und filtere ...")
    table = corpus_module.read_metadata(raw)
    n = args.n or int(config.corpus["target_n"])
    selection, stats = corpus_module.build_corpus(
        table,
        n=n,
        label_field=config.corpus["label_field"],
        seed=args.seed,
        min_class_size=int(config.corpus.get("min_class_size", 20)),
        facet_cap=int(config.corpus.get("facet_cap", 16)),
    )

    path = corpus_module.write_corpus(selection, run / "corpus.parquet")
    (run / "corpus_stats.json").write_text(
        json.dumps(
            {
                "total": stats.total,
                "public_domain": stats.public_domain,
                "with_title": stats.with_title,
                "after_dedup": stats.after_dedup,
                "selected": stats.selected,
                "label_classes": stats.label_classes,
                "dropped_classes": stats.dropped_classes,
                "facet_sizes": stats.facet_sizes,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    _section("Auswahl")
    _ok("Objekte gesamt", f"{stats.total:,}")
    _ok("davon gemeinfrei", f"{stats.public_domain:,}")
    _ok("mit Titel", f"{stats.with_title:,}")
    _ok("nach Entdopplung", f"{stats.after_dedup:,}")
    _ok("gewaehlt", f"{stats.selected:,}")
    _ok("Labelklassen", stats.label_classes)
    _ok("Facettengroessen", stats.facet_sizes)
    if stats.dropped_classes:
        _ok("verworfene Klassen", f"{len(stats.dropped_classes)} (zu klein oder Rang > Limit)")
    print(f"\n  -> {path.relative_to(config.root)}\n")
    return 0


# ---------------------------------------------------------------------------
# images / embed  (lokal)
# ---------------------------------------------------------------------------


def cmd_images(args: argparse.Namespace) -> int:
    from . import corpus as corpus_module, images as images_module

    config = load()
    run = _resolve(config, args.out)
    table = corpus_module.read_corpus(run / "corpus.parquet")
    object_ids = table.column("object_id").to_pylist()

    print(f"Bilder fuer {len(object_ids):,} Objekte -> {run.name}/images")
    stats = images_module.fetch_images(
        object_ids,
        run / "images",
        api=config.corpus["object_api"],
        size=args.size,
        workers=args.workers,
        progress=_bar,
    )
    _section("Bilder")
    _ok("angefragt", f"{stats.requested:,}")
    _ok("uebersprungen", f"{stats.skipped:,} (bereits im Journal)")
    _ok("heruntergeladen", f"{stats.downloaded:,}")
    _ok("ohne Bild", f"{stats.resolved - stats.downloaded:,}")
    if stats.failed:
        _bad("Fehler", f"{stats.failed:,}")
    _ok("Dauer", f"{stats.seconds:.0f}s")
    print()
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    from . import corpus as corpus_module, embedding, images as images_module

    config = load()
    run = _resolve(config, args.out)
    table = corpus_module.read_corpus(run / "corpus.parquet")
    object_ids = table.column("object_id").to_pylist()

    indices, paths = images_module.available_images(run / "images", object_ids)
    if not paths:
        _bad("Bilder", f"keine in {run / 'images'} — zuerst `lse images` ausfuehren")
        return 1
    print(f"Bette {len(paths):,} von {len(object_ids):,} Objekten ein "
          f"({len(object_ids) - len(paths):,} ohne Bild)")

    embeddings, stats = embedding.embed_images(
        paths,
        model_id=config.model["id"],
        revision=config.model["revision"],
        batch_size=int(config.model["batch_size"]),
        dtype=config.model["dtype"],
        device=args.device,
        shard_dir=run / "shards",
        num_workers=int(config.model.get("num_workers", 8)),
        progress=_bar,
    )

    np.save(run / "embeddings.f32.npy", embeddings, allow_pickle=False)
    # Die Zeilen, die tatsaechlich eingebettet wurden. Ohne diese Datei laesst
    # sich der Korpus spaeter nicht auf dieselbe Menge reduzieren — und genau
    # dort bricht sonst die ID-Invariante.
    np.save(run / "embedded_rows.i64.npy", np.asarray(indices, dtype=np.int64))

    _section("Embeddings")
    _ok("n x dim", f"{stats.n:,} x {stats.dim}")
    _ok("Geraet", stats.device)
    _ok("Dauer", f"{stats.seconds:.0f}s ({stats.n / max(stats.seconds, 1e-6):.0f} Bilder/s)")
    print()
    return 0


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------


def cmd_project(args: argparse.Namespace) -> int:
    from . import corpus as corpus_module, projection, synthetic, validation
    from .artifacts import ArtifactWriter

    config = load()
    run = _resolve(config, args.out)
    table = corpus_module.read_corpus(run / "corpus.parquet")

    embeddings_path = run / "embeddings.f32.npy"
    rows_path = run / "embedded_rows.i64.npy"
    placeholder = False

    if embeddings_path.is_file():
        embeddings = np.load(embeddings_path)
        if rows_path.is_file():
            keep = np.load(rows_path)
            table = table.take(keep)
    elif args.placeholder:
        # Ausdruecklich markierte Platzhalter: erlaubt, den Viewer und die
        # Interaktion gegen ECHTE Metadaten zu bauen, solange der Einbettungs-
        # schritt nicht laufen kann. Der Raum bedeutet dabei nichts, und das
        # Manifest sagt das auch.
        placeholder = True
        embeddings, _, _ = synthetic.make_embeddings(
            table.num_rows, dim=int(config.model["dim"]), n_clusters=8, seed=args.seed
        )
        print("\033[33m  Platzhalter-Embeddings — die Geometrie ist bedeutungslos.\033[0m")
    else:
        _bad("Embeddings", f"{embeddings_path} fehlt — `lse embed` ausfuehren")
        print("  Oder mit --placeholder einen bedeutungslosen, klar markierten Raum bauen.")
        return 1

    labels_text = table.column(config.corpus["label_field"]).to_pylist()
    labels, label_names = corpus_module.label_encode(labels_text)

    params = dict(config.projection)
    if args.epochs:
        params["umap_n_epochs"] = args.epochs

    print(f"Projiziere {len(embeddings):,} x {embeddings.shape[1]} ...")
    result, pca, _ = projection.project(embeddings, params, verbose=True)
    projection.save_pca(pca, run / projection.PCA_FILE)
    _ok("Dauer", f"{result.seconds:.0f}s")

    gate_result = None
    if not args.no_gate:
        print("\nShuffle-Control ...")
        shuffled = validation.shuffle_columns(embeddings, seed=args.seed)
        shuffled_result, _, _ = projection.project(shuffled, params, keep_knn=False)

        gate_result = validation.run_gate(
            embeddings,
            result.coords,
            labels,
            k=int(config.validation["knn_k"]),
            shuffled_coords=shuffled_result.coords,
            shuffled_embeddings=shuffled,
            min_knn_overlap=float(config.validation["min_knn_overlap"]),
            high_knn=result.knn_indices,
        )
        _section("Go/No-Go-Tor")
        print(gate_result.summary())
        plot = validation.plot_gate(
            result.coords,
            labels,
            label_names,
            run / "gate.png",
            shuffled_coords=shuffled_result.coords,
            title=f"{config.corpus['id']} — n={len(embeddings):,}"
            + (" — PLATZHALTER" if placeholder else ""),
        )
        (run / "gate.json").write_text(
            json.dumps(gate_result.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print()
        if gate_result.passed:
            _ok("Urteil", "\033[32mBESTANDEN\033[0m")
        else:
            _bad("Urteil", "\033[31mDURCHGEFALLEN\033[0m")
            for reason in gate_result.reasons:
                print(f"      {reason}")
            print("\n  Wenn bekannte Labels sich nicht trennen: Korpus wechseln,")
            print("  nicht Parameter tunen (siehe docs/decisions.md E6).")
        _ok("Plot", plot.relative_to(config.root))

    neighbors_k = int(config.artifacts["neighbors_k"])
    neighbors = (
        result.knn_indices[:, :neighbors_k]
        if result.knn_indices is not None and result.knn_indices.shape[1] >= neighbors_k
        else validation.knn_indices_high(embeddings, neighbors_k)
    )

    writer = ArtifactWriter(run, config)
    spec = writer.add_coords(result.coords)
    writer.add_labels(labels, label_names)
    writer.add_colors(synthetic.palette_colors(labels))
    writer.add_neighbors(np.asarray(neighbors, dtype=np.uint32))
    if args.keep_embeddings:
        writer.add_embeddings(embeddings)

    manifest = writer.finish(
        corpus={
            "id": config.corpus["id"],
            "label_field": config.corpus["label_field"],
            "facets": list(config.corpus["facets"]),
            "seed": args.seed,
        },
        model=(
            {"id": "placeholder", "revision": "none", "dim": int(embeddings.shape[1])}
            if placeholder
            else {
                "id": config.model["id"],
                "revision": config.model["revision"],
                "dim": int(embeddings.shape[1]),
                "dtype": config.model["dtype"],
            }
        ),
        projection={
            **params,
            "pca_components": result.pca_components,
            "pca_explained_variance": result.explained_variance,
            "seconds": result.seconds,
            "placeholder_embeddings": placeholder,
        },
        validation=gate_result.to_dict() if gate_result else {},
    )

    _section("Artefakte")
    _ok("n", f"{manifest.n:,}")
    _ok("Radius", f"{spec.radius:.1f} (Box +/-{spec.box:g})")
    _ok("Groesse", f"{sum(f['bytes'] for f in manifest.files) / 1024:.0f} KiB")
    print(f"\n  -> {run.relative_to(config.root)}\n")
    return 0 if (gate_result is None or gate_result.passed) else 2


# ---------------------------------------------------------------------------
# synth / verify
# ---------------------------------------------------------------------------


def cmd_synth(args: argparse.Namespace) -> int:
    from . import projection, synthetic
    from .artifacts import ArtifactWriter

    config = load()
    out = _resolve(config, args.out)
    k = int(config.artifacts["neighbors_k"])
    print(f"Erzeuge {args.n:,} synthetische Punkte ({args.mode}) nach {out} ...")

    if args.mode == "umap":
        embeddings, labels, names = synthetic.make_embeddings(
            args.n, dim=int(config.model["dim"]), n_clusters=args.clusters, seed=args.seed
        )
        result, _, _ = projection.project(embeddings, dict(config.projection), verbose=True)
        coords = result.coords
        neighbors = synthetic.high_dim_neighbors(embeddings, k)
    else:
        coords, labels, names = synthetic.make_coords(
            args.n, n_clusters=args.clusters, seed=args.seed
        )
        embeddings = None
        neighbors = np.tile(np.arange(min(k, args.n), dtype=np.uint32), (args.n, 1))

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
            "note": "Synthetische Gauss-Blobs — kein echter Korpus.",
            "n_clusters": args.clusters,
            "seed": args.seed,
        },
        model={"id": "none", "revision": "none", "dim": int(config.model["dim"])},
        projection=(
            dict(config.projection) if args.mode == "umap" else {"method": "direct-3d-blobs"}
        ),
    )
    _ok("n", f"{manifest.n:,}")
    _ok("Radius", f"{spec.radius:.1f} (Box +/-{spec.box:g})")
    _ok("Groesse", f"{sum(f['bytes'] for f in manifest.files) / 1024:.1f} KiB")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .artifacts import ArtifactError, load_manifest, verify

    config = load()
    target = _resolve(config, args.path)
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
    _ok("Modell", f"{manifest.model.get('id','?')} @ {manifest.model.get('revision','?')}")
    _ok("Commit", (manifest.provenance.get("git_commit") or "—")[:12])
    if manifest.projection.get("placeholder_embeddings"):
        _bad("Achtung", "Platzhalter-Embeddings — die Geometrie bedeutet nichts")
    if manifest.validation:
        _ok("Tor", "bestanden" if manifest.validation.get("passed") else "DURCHGEFALLEN")

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
    doctor.add_argument("--network", action="store_true", help="Hosts gegen die Egress-Richtlinie pruefen")
    doctor.set_defaults(func=cmd_doctor)

    fetch = sub.add_parser("fetch", help="Metadaten laden und Auswahl bauen")
    fetch.add_argument("--n", type=int, default=None, help="Anzahl Objekte (Vorgabe: corpus.target_n)")
    fetch.add_argument("--seed", type=int, default=20260803)
    fetch.add_argument("--force", action="store_true", help="CSV neu herunterladen")
    fetch.add_argument("--out", default=DEFAULT_RUN)
    fetch.set_defaults(func=cmd_fetch)

    images = sub.add_parser("images", help="Bilder holen und verkleinern (lokal)")
    images.add_argument("--size", type=int, default=224)
    images.add_argument("--workers", type=int, default=8)
    images.add_argument("--out", default=DEFAULT_RUN)
    images.set_defaults(func=cmd_images)

    embed = sub.add_parser("embed", help="Bilder mit SigLIP 2 einbetten (lokal)")
    embed.add_argument("--device", default=None, help="cuda / cpu (Vorgabe: automatisch)")
    embed.add_argument("--out", default=DEFAULT_RUN)
    embed.set_defaults(func=cmd_embed)

    project = sub.add_parser("project", help="Projektion, Go/No-Go-Tor und Artefakte")
    project.add_argument("--seed", type=int, default=20260803)
    project.add_argument("--epochs", type=int, default=None, help="UMAP-Epochen ueberschreiben")
    project.add_argument("--no-gate", action="store_true", help="Validierung ueberspringen")
    project.add_argument(
        "--placeholder",
        action="store_true",
        help="ohne Embeddings arbeiten: bedeutungslose, klar markierte Geometrie",
    )
    project.add_argument("--keep-embeddings", action="store_true")
    project.add_argument("--out", default=DEFAULT_RUN)
    project.set_defaults(func=cmd_project)

    synth = sub.add_parser("synth", help="synthetischen Artefaktsatz erzeugen")
    synth.add_argument("--n", type=int, default=1000)
    synth.add_argument("--clusters", type=int, default=8)
    synth.add_argument("--seed", type=int, default=0)
    synth.add_argument("--mode", choices=("coords", "umap"), default="coords")
    synth.add_argument("--keep-embeddings", action="store_true")
    synth.add_argument("--out", default="data/sample")
    synth.set_defaults(func=cmd_synth)

    verify_cmd = sub.add_parser("verify", help="Artefaktsatz gegen sein Manifest pruefen")
    verify_cmd.add_argument("path", help="Verzeichnis mit manifest.json")
    verify_cmd.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"\033[31mKonfigurationsfehler:\033[0m {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
