"""Der Datenvertrag zwischen ``pipeline/`` und ``viewer/``.

Die beiden Haelften des Projekts kommunizieren ausschliesslich ueber Dateien auf
der Platte — keine Imports, kein laufender Prozess dazwischen. Dieses Modul ist
die Python-Seite des Vertrags; ``viewer/src/artifacts.js`` ist die andere.
**Aenderungen hier muessen dort nachgezogen werden**, und beide Seiten gehoeren
in dasselbe Review.

Zwei Invarianten, die projektweit gelten:

1. **Der Zeilenindex ist die ID.** Korpus, Embeddings, Koordinaten, Farben,
   Nachbarn und Thumbnails werden in identischer Reihenfolge geschrieben.
   :func:`ArtifactWriter.finish` prueft das und verweigert sonst den Dienst.
2. **Koordinaten sind normalisiert.** Der UMAP-Rohausgabemassstab ist willkuerlich
   und aendert sich bei jedem Neulauf. Ohne Normalisierung entwertet jeder Lauf
   still jede Kamerakonstante, Nebeldichte und Punktgroesse im Viewer.

Alle Binaerdateien sind headerlos und explizit little-endian, damit die
Browserseite ``new Int16Array(await res.arrayBuffer())`` ohne Parsing und ohne
Kopie machen kann.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import SCHEMA_VERSION, Config

MANIFEST_NAME = "manifest.json"

# int16-SNORM: WebGL bildet einen signed-16-Bit-Wert mit ``normalized: true`` auf
# ``max(q / 32767, -1)`` ab. Dieselbe Konstante auf beiden Seiten.
I16_SCALE = 32767.0

# Koordinaten und Farben werden vierkomponentig geschrieben, obwohl nur drei
# Komponenten Bedeutung tragen.
#
# Das ist keine Bequemlichkeit, sondern eine Anforderung der Zielplattform:
# **WebGPU kennt keine dreikomponentigen 8- und 16-Bit-Vertexformate.** Es gibt
# ``snorm16x2`` und ``snorm16x4``, aber kein ``snorm16x3``; ebenso ``unorm8x2``
# und ``unorm8x4``, aber kein ``unorm8x3``. Nur ``float32`` gibt es dreiweit.
#
# Ein int16x3-Attribut muesste also beim Laden im Browser einmal ueber alle
# Punkte umkopiert werden — bei 1 Mio. Punkten genau die Art Arbeit, die der
# binaere, zero-copy Vertrag vermeiden soll. Der Preis sind 8 statt 6 Byte pro
# Punkt (800 KB statt 600 KB bei 100k) gegen float32 mit 1,2 MB und JSON mit
# 18 MB. Die vierte Komponente ist reserviert.
COORD_STRIDE = 4
COLOR_STRIDE = 4

# Dateinamen des Vertrags. Der Viewer kennt exakt diese Namen.
COORDS_FILE = "coords.i16.bin"
COLORS_FILE = "colors.u8.bin"
LABELS_FILE = "labels.u16.bin"
NEIGHBORS_FILE = "neighbors.u32.bin"
EMBEDDINGS_FILE = "embeddings.f32.npy"
CORPUS_FILE = "corpus.parquet"
FACETS_FILE = "facets.u8.bin"
TEXT_FILE = "text.bin"
TEXT_OFFSETS_FILE = "text_offsets.u32.bin"

# Trennzeichen innerhalb eines Datensatzes (ASCII 31, Unit Separator) und
# zwischen Datensaetzen (ASCII 30, Record Separator). Beide sind genau dafuer
# gedacht und koennen in Museumsmetadaten nicht vorkommen — anders als
# Semikolon, Pipe oder Tab, die alle in echten Titeln auftreten.
#
# Der Datensatz-Trenner ist nicht redundant zur Offsettabelle: mit ihm kann der
# Browser den gesamten Block EINMAL dekodieren und darin suchen, statt 100.000
# Einzelaufrufe von TextDecoder zu machen.
FIELD_SEPARATOR = "\x1f"
RECORD_SEPARATOR = "\x1e"


class ArtifactError(RuntimeError):
    """Ein Artefaktsatz ist unvollstaendig oder verletzt eine Invariante."""


# ---------------------------------------------------------------------------
# Koordinaten: normalisieren und quantisieren
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoordSpec:
    """Alles, was der Viewer braucht, um ``coords.i16.bin`` zu dequantisieren.

    Die Dequantisierung im Vertex-Shader ist
    ``pos = q * half + offset`` mit ``q`` aus einem normalisierten
    Int16-Attribut (also bereits im Bereich [-1, 1]).
    """

    offset: tuple[float, float, float]
    half: tuple[float, float, float]
    # Beschreibt die normalisierte Wolke; der Viewer leitet daraus Kamera,
    # Near/Far und Nebeldichte ab, statt sie hart zu verdrahten.
    min: tuple[float, float, float]
    max: tuple[float, float, float]
    radius: float
    # Herkunft: wie aus dem UMAP-Rohausgang die normalisierte Wolke wurde.
    source_centroid: tuple[float, float, float]
    source_scale: float
    box: float


def normalize_coords(coords: np.ndarray, box: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Zentriert die Wolke und skaliert sie **uniform** in eine Box ``+/- box``.

    Uniform ist wesentlich: eine Skalierung pro Achse wuerde die Projektion
    verzerren und damit genau die Nachbarschaftsbeziehungen kaputtmachen, die
    das Bild aussagen soll.

    Rueckgabe: normalisierte Koordinaten, Schwerpunkt des Originals, Skalenfaktor.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ArtifactError(f"Koordinaten muessen die Form (N, 3) haben, nicht {coords.shape}.")
    if len(coords) == 0:
        raise ArtifactError("Leerer Koordinatensatz.")

    centroid = coords.mean(axis=0)
    centered = coords - centroid

    max_abs = float(np.abs(centered).max())
    if max_abs == 0.0:
        # Alle Punkte identisch. Pathologisch, aber kein Grund zum Absturz —
        # das faellt im Go/No-Go-Tor auf, nicht hier.
        scale = 1.0
    else:
        scale = box / max_abs

    return centered * scale, centroid, scale


def quantize_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Bildet jede Achse einzeln auf den vollen int16-Bereich ab.

    Pro Achse statt global, weil das die Aufloesung maximiert: eine flache Wolke
    verschenkt sonst eine ganze Achse. Die Dequantisierung stellt die Werte
    exakt wieder her, die Form bleibt also unangetastet.
    """
    coords = np.asarray(coords, dtype=np.float64)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)

    offset = (hi + lo) / 2.0
    half = (hi - lo) / 2.0
    # Entartete Achse (alle Werte gleich): half=0 wuerde durch null teilen.
    half = np.where(half == 0.0, 1.0, half)

    q = np.rint((coords - offset) / half * I16_SCALE)
    q = np.clip(q, -I16_SCALE, I16_SCALE).astype("<i2")
    return q, (offset, half)


def dequantize_coords(q: np.ndarray, spec: CoordSpec) -> np.ndarray:
    """Umkehrung von :func:`quantize_coords` — dieselbe Rechnung wie im Shader.

    Nimmt sowohl die dreikomponentige Form als auch die vierkomponentige an, die
    auf der Platte liegt (siehe :data:`COORD_STRIDE`).
    """
    q = np.asarray(q)
    if q.ndim == 2 and q.shape[1] == COORD_STRIDE:
        q = q[:, :3]
    offset = np.asarray(spec.offset, dtype=np.float64)
    half = np.asarray(spec.half, dtype=np.float64)
    return q.astype(np.float64) / I16_SCALE * half + offset


def _pad_to_stride(arr: np.ndarray, stride: int, fill: int = 0) -> np.ndarray:
    """Erweitert ein (N, 3)-Array auf (N, stride); siehe :data:`COORD_STRIDE`."""
    if arr.ndim != 2:
        raise ArtifactError(f"Erwartet ein zweidimensionales Array, nicht {arr.shape}.")
    if arr.shape[1] == stride:
        return arr
    if arr.shape[1] != 3:
        raise ArtifactError(f"Erwartet 3 oder {stride} Spalten, nicht {arr.shape[1]}.")
    padded = np.full((len(arr), stride), fill, dtype=arr.dtype)
    padded[:, :3] = arr
    return padded


def prepare_coords(coords: np.ndarray, box: float) -> tuple[np.ndarray, CoordSpec]:
    """Normalisieren und quantisieren in einem Schritt."""
    normalized, centroid, scale = normalize_coords(coords, box)
    q, (offset, half) = quantize_coords(normalized)

    radius = float(np.linalg.norm(normalized, axis=1).max())
    return q, CoordSpec(
        offset=tuple(float(v) for v in offset),
        half=tuple(float(v) for v in half),
        min=tuple(float(v) for v in normalized.min(axis=0)),
        max=tuple(float(v) for v in normalized.max(axis=0)),
        radius=radius,
        source_centroid=tuple(float(v) for v in centroid),
        source_scale=float(scale),
        box=float(box),
    )


# ---------------------------------------------------------------------------
# Dateien
# ---------------------------------------------------------------------------


# Ab dieser Ersparnis lohnt die komprimierte Zweitfassung. Darunter kostet der
# zusaetzliche Abruf mehr, als er spart.
GZIP_MIN_SAVING = 0.15
# Unterhalb dieser Groesse dominiert der Verbindungsaufbau, nicht die Nutzlast.
GZIP_MIN_BYTES = 64 * 1024


@dataclass
class FileEntry:
    """Ein Eintrag in der Dateiliste des Manifests."""

    name: str
    dtype: str
    shape: list[int]
    bytes: int
    sha256: str
    # Komprimierte Zweitfassung, falls sie sich lohnt.
    #
    # Sie liegt NICHT deshalb da, weil HTTP nicht komprimieren koennte, sondern
    # weil statische Hosts nach Content-Type entscheiden — und
    # `application/octet-stream` ist bei GitHub Pages und Verwandten meist
    # nicht dabei. Ohne eigene Fassung wuerden bei 100k Punkten 12,5 MB Text
    # statt 4,9 MB uebertragen, und Header lassen sich dort nicht setzen.
    # Der Browser packt sie mit `DecompressionStream` selbst aus.
    gzip_bytes: int | None = None
    gzip_sha256: str | None = None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def maybe_compress(path: Path, entry: FileEntry) -> FileEntry:
    """Legt eine gzip-Fassung daneben, wenn sie sich lohnt.

    Wird sie kleiner als der Schwellwert hergibt, bleibt es bei der rohen
    Datei und eine eventuell vorhandene alte ``.gz`` wird entfernt — sonst
    laedt der Viewer eine veraltete Zweitfassung.
    """
    target = path.with_suffix(path.suffix + ".gz")
    if entry.bytes < GZIP_MIN_BYTES:
        target.unlink(missing_ok=True)
        return entry

    raw = path.read_bytes()
    # mtime=0: sonst unterscheiden sich zwei Laeufe derselben Daten in der
    # Pruefsumme, und "muss ich neu generieren?" wird unbeantwortbar.
    packed = gzip.compress(raw, compresslevel=6, mtime=0)
    if len(packed) > entry.bytes * (1 - GZIP_MIN_SAVING):
        target.unlink(missing_ok=True)
        return entry

    target.write_bytes(packed)
    entry.gzip_bytes = len(packed)
    entry.gzip_sha256 = hashlib.sha256(packed).hexdigest()
    return entry


def write_binary(path: Path, arr: np.ndarray, dtype: str) -> FileEntry:
    """Schreibt ein headerloses, little-endian Binaerarray.

    Explizit little-endian statt Plattformordnung: der Browser liest den Puffer
    ohne Interpretation, ein Big-Endian-Build wuerde still Unsinn erzeugen.
    """
    arr = np.ascontiguousarray(arr, dtype=np.dtype(dtype))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(arr.tobytes(order="C"))
    return maybe_compress(path, FileEntry(
        name=path.name,
        dtype=dtype,
        shape=list(arr.shape),
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
    ))


def read_binary(path: Path, dtype: str, shape: Iterable[int]) -> np.ndarray:
    shape = tuple(shape)
    arr = np.fromfile(path, dtype=np.dtype(dtype))
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ArtifactError(
            f"{path}: {arr.size} Elemente gelesen, {expected} erwartet (Form {shape})."
        )
    return arr.reshape(shape)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class Manifest:
    """Das Manifest neben jedem Artefaktsatz.

    Zweck: aus "muss ich das neu generieren?" eine Frage mit Antwort machen.
    Es haelt fest, welches Modell, welche Parameter, welche Seeds und welcher
    Commit die Dateien erzeugt haben — inklusive Bibliotheksversionen, weil
    UMAP-Pickles ueber numpy-/numba-Wechsel brechen.
    """

    schema_version: int
    created_at: str
    n: int
    corpus: dict[str, Any]
    model: dict[str, Any]
    projection: dict[str, Any]
    coords: dict[str, Any]
    validation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "Manifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        found = data.get("schema_version")
        if found != SCHEMA_VERSION:
            raise ArtifactError(
                f"{path}: schema_version {found!r}, erwartet {SCHEMA_VERSION}."
            )
        return cls(**data)


def git_commit(root: Path) -> str | None:
    """Der Commit, aus dem der Lauf stammt — oder None ausserhalb eines Repos."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def library_versions() -> dict[str, str]:
    """Versionen der Bibliotheken, die das Ergebnis beeinflussen."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("numpy", "sklearn", "umap", "numba", "pynndescent", "pyarrow"):
        try:
            module = __import__(name)
        except Exception:  # pragma: no cover - haengt von der Installation ab
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


# ---------------------------------------------------------------------------
# Schreiben eines vollstaendigen Artefaktsatzes
# ---------------------------------------------------------------------------


class ArtifactWriter:
    """Schreibt einen Artefaktsatz und erzwingt dabei die ID-Invariante.

    Benutzung::

        writer = ArtifactWriter(out_dir, config)
        writer.add_coords(coords_3d)
        writer.add_labels(labels, names)
        writer.add_colors(colors)
        writer.finish(corpus={...}, model={...}, projection={...})

    ``finish`` verweigert den Dienst, wenn die Zeilenzahlen auseinanderlaufen —
    das ist der Fehler, der sonst erst im Browser auffaellt, als stille
    Fehlzuordnung zwischen Punkt und Metadatum.
    """

    def __init__(self, out_dir: Path, config: Config) -> None:
        self.dir = Path(out_dir)
        self.config = config
        self.dir.mkdir(parents=True, exist_ok=True)
        self._files: list[FileEntry] = []
        self._counts: dict[str, int] = {}
        self._coord_spec: CoordSpec | None = None
        self._label_names: list[str] = []
        self._metadata: dict[str, Any] = {}

    # -- Bestandteile ----------------------------------------------------

    def add_coords(self, coords: np.ndarray) -> CoordSpec:
        box = float(self.config.artifacts["coord_box"])
        q, spec = prepare_coords(coords, box)
        entry = write_binary(self.dir / COORDS_FILE, _pad_to_stride(q, COORD_STRIDE), "<i2")
        self._files.append(entry)
        self._counts["coords"] = len(q)
        self._coord_spec = spec
        return spec

    def add_labels(self, labels: np.ndarray, names: list[str]) -> None:
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ArtifactError(f"Labels muessen eindimensional sein, nicht {labels.shape}.")
        if len(names) > 0xFFFF:
            raise ArtifactError(f"{len(names)} Labelklassen sprengen den uint16-Bereich.")
        if labels.size and int(labels.max()) >= len(names):
            raise ArtifactError(
                f"Label-Index {int(labels.max())} hat keinen Namen ({len(names)} Namen bekannt)."
            )
        entry = write_binary(self.dir / LABELS_FILE, labels, "<u2")
        self._files.append(entry)
        self._counts["labels"] = len(labels)
        self._label_names = list(names)

    def add_colors(self, colors: np.ndarray) -> None:
        colors = np.asarray(colors)
        if colors.ndim != 2 or colors.shape[1] not in (3, COLOR_STRIDE):
            raise ArtifactError(f"Farben muessen die Form (N, 3) haben, nicht {colors.shape}.")
        # Vierte Komponente auf 255: als Alpha gelesen ist das "voll sichtbar",
        # und der Kanal steht spaeter fuer ein Per-Punkt-Flag zur Verfuegung.
        padded = _pad_to_stride(colors.astype(np.uint8), COLOR_STRIDE, fill=255)
        entry = write_binary(self.dir / COLORS_FILE, padded, "<u1")
        self._files.append(entry)
        self._counts["colors"] = len(colors)

    def add_neighbors(self, neighbors: np.ndarray) -> None:
        """Echte Top-k-Nachbarn aus dem **hochdimensionalen** Raum.

        Nicht aus den 3D-Koordinaten: gerade die Abweichung zwischen beidem ist
        die Aussage (siehe Ehrlichkeitsschicht in docs/review-2026-08-03.md).
        """
        neighbors = np.asarray(neighbors)
        if neighbors.ndim != 2:
            raise ArtifactError(f"Nachbarn muessen die Form (N, k) haben, nicht {neighbors.shape}.")
        entry = write_binary(self.dir / NEIGHBORS_FILE, neighbors, "<u4")
        self._files.append(entry)
        self._counts["neighbors"] = len(neighbors)

    def add_metadata(
        self,
        rows: list[dict[str, Any]],
        *,
        facet_fields: list[str],
        text_fields: list[str],
    ) -> dict[str, Any]:
        """Schreibt Facettenindizes und die Textfelder fuer das Detailpanel.

        Zwei getrennte Formate, weil die Zugriffsmuster verschieden sind:

        * **Facetten** werden bei jedem Filterklick ueber alle Punkte gelesen und
          liegen deshalb als dichtes ``uint8``-Array vor — ein Index je Punkt
          und Facette. Bei 100k Punkten und vier Facetten sind das 400 KB, die
          der Browser ohne Parsen uebernimmt.
        * **Text** wird immer nur fuer einen einzelnen Punkt gebraucht und liegt
          deshalb als ein einziger UTF-8-Block mit Offsettabelle vor. 100.000
          JSON-Objekte zu parsen kostet hunderte Millisekunden Hauptthread und
          erzeugt 100.000 kurzlebige JS-Objekte; ein Block plus Offsets kostet
          nichts und wird nachgeladen, nicht mitgeladen.
        """
        if len(facet_fields) > 255:
            raise ArtifactError("Mehr als 255 Facetten sind nicht vorgesehen.")

        # Facettenvokabulare: der Index in dieser Liste steht in der Binaerdatei.
        vocabularies: dict[str, list[str]] = {}
        for field in facet_fields:
            values = sorted({str(row.get(field, "")) for row in rows})
            if len(values) > 255:
                raise ArtifactError(
                    f"Facette {field!r} hat {len(values)} Werte — mehr als uint8 traegt. "
                    "Im Korpusschritt kappen (siehe lse.corpus.cap_facet)."
                )
            vocabularies[field] = values

        lookups = {
            field: {value: index for index, value in enumerate(values)}
            for field, values in vocabularies.items()
        }
        facet_matrix = np.array(
            [[lookups[field][str(row.get(field, ""))] for field in facet_fields] for row in rows],
            dtype=np.uint8,
        ).reshape(len(rows), len(facet_fields))
        self._files.append(write_binary(self.dir / FACETS_FILE, facet_matrix, "<u1"))

        # Textblock plus Offsets. Die Offsets sind N+1 Eintraege, damit die
        # Laenge des letzten Datensatzes ohne Sonderfall ableitbar ist.
        blob = bytearray()
        offsets = np.zeros(len(rows) + 1, dtype=np.uint32)
        for index, row in enumerate(rows):
            record = FIELD_SEPARATOR.join(str(row.get(field, "")) for field in text_fields)
            blob.extend(record.encode("utf-8"))
            blob.extend(RECORD_SEPARATOR.encode("utf-8"))
            offsets[index + 1] = len(blob)

        text_path = self.dir / TEXT_FILE
        text_path.write_bytes(bytes(blob))
        self._files.append(
            maybe_compress(
                text_path,
                FileEntry(
                    name=TEXT_FILE,
                    dtype="utf-8",
                    shape=[len(rows)],
                    bytes=text_path.stat().st_size,
                    sha256=sha256_file(text_path),
                ),
            )
        )
        self._files.append(write_binary(self.dir / TEXT_OFFSETS_FILE, offsets, "<u4"))

        self._counts["facets"] = len(rows)
        self._metadata = {
            "facet_fields": list(facet_fields),
            "facet_values": vocabularies,
            "text_fields": list(text_fields),
            "field_separator": FIELD_SEPARATOR,
            "record_separator": RECORD_SEPARATOR,
        }
        return self._metadata

    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """Hochdimensionale Embeddings als ``.npy``.

        ``.npy`` und nicht ``.npz`` (Zip, nicht memmapbar) und niemals Pickle.
        Bleibt auf der Python-Seite; der Browser bekommt sie nie zu sehen.
        """
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        path = self.dir / EMBEDDINGS_FILE
        np.save(path, embeddings, allow_pickle=False)
        self._files.append(
            FileEntry(
                name=path.name,
                dtype="<f4",
                shape=list(embeddings.shape),
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
        self._counts["embeddings"] = len(embeddings)

    # -- Abschluss -------------------------------------------------------

    def finish(
        self,
        *,
        corpus: dict[str, Any] | None = None,
        model: dict[str, Any] | None = None,
        projection: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
    ) -> Manifest:
        if self._coord_spec is None:
            raise ArtifactError("Kein Koordinatensatz geschrieben — add_coords fehlt.")

        counts = set(self._counts.values())
        if len(counts) != 1:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(self._counts.items()))
            raise ArtifactError(
                "ID-Invariante verletzt: unterschiedliche Zeilenzahlen "
                f"({detail}). Der Zeilenindex ist die ID — alle Artefakte muessen "
                "in identischer Reihenfolge und Laenge geschrieben werden."
            )
        n = counts.pop()

        coords_section = asdict(self._coord_spec)
        coords_section["i16_scale"] = I16_SCALE
        # Der Viewer liest die Schrittweiten aus dem Manifest, statt sie zu
        # kennen — sonst waere eine Formataenderung eine stille Fehlinterpretation
        # statt eines lauten Fehlers.
        coords_section["coord_stride"] = COORD_STRIDE
        coords_section["color_stride"] = COLOR_STRIDE

        manifest = Manifest(
            schema_version=SCHEMA_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n=n,
            corpus={**(corpus or {}), "label_names": self._label_names},
            model=model or {},
            projection=projection or {},
            coords=coords_section,
            validation=validation or {},
            metadata=self._metadata,
            files=[asdict(f) for f in self._files],
            provenance={
                "git_commit": git_commit(self.config.root),
                "libraries": library_versions(),
                "config": self.config.data,
            },
        )
        manifest.write(self.dir / MANIFEST_NAME)
        return manifest


def load_manifest(run_dir: Path) -> Manifest:
    path = Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        raise ArtifactError(f"{path} fehlt — kein Artefaktsatz in {run_dir}.")
    return Manifest.read(path)


def verify(run_dir: Path) -> list[str]:
    """Prueft einen Artefaktsatz gegen sein Manifest.

    Rueckgabe ist eine Liste von Problemen; leer heisst in Ordnung.
    """
    run_dir = Path(run_dir)
    problems: list[str] = []
    manifest = load_manifest(run_dir)

    for entry in manifest.files:
        path = run_dir / entry["name"]
        if not path.is_file():
            problems.append(f"{entry['name']}: fehlt")
            continue
        size = path.stat().st_size
        if size != entry["bytes"]:
            problems.append(f"{entry['name']}: {size} Bytes, {entry['bytes']} laut Manifest")
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            problems.append(f"{entry['name']}: Pruefsumme weicht ab")
            continue

        if entry.get("gzip_bytes"):
            packed = run_dir / (entry["name"] + ".gz")
            if not packed.is_file():
                problems.append(f"{entry['name']}.gz: fehlt")
            elif packed.stat().st_size != entry["gzip_bytes"]:
                problems.append(f"{entry['name']}.gz: Groesse weicht ab")
            elif sha256_file(packed) != entry["gzip_sha256"]:
                problems.append(f"{entry['name']}.gz: Pruefsumme weicht ab")

    for entry in manifest.files:
        shape = entry.get("shape") or []
        if entry["name"] == EMBEDDINGS_FILE:
            continue
        # Die Offsettabelle hat bewusst N+1 Eintraege: so ist die Laenge des
        # letzten Datensatzes ohne Sonderfall ableitbar.
        expected = manifest.n + 1 if entry["name"] == TEXT_OFFSETS_FILE else manifest.n
        if shape and shape[0] != expected:
            problems.append(
                f"{entry['name']}: {shape[0]} Zeilen, Manifest sagt n={manifest.n} "
                "(ID-Invariante)"
            )

    return problems
