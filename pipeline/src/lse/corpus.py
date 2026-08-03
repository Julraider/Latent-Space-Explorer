"""Der Korpus: Met Open Access, von der Roh-CSV zur Auswahl.

Zustaendig fuer alles, was vor dem Modell passiert — herunterladen, filtern,
entdoppeln, Facetten normalisieren, Stichprobe ziehen. Das Ergebnis ist
``corpus.parquet``, und dessen **Zeilenreihenfolge ist ab da bindend**: alle
weiteren Artefakte (Embeddings, Koordinaten, Farben, Nachbarn) werden in
identischer Reihenfolge geschrieben (siehe :mod:`lse.artifacts`).

Zwei Dinge, die die CSV *nicht* enthaelt und die deshalb spaeter kommen:

* **Bild-URLs.** Die CSV kennt nur ``Link Resource``, die Museumsseite. Die
  eigentliche Bilddatei steht in der Objekt-API unter ``primaryImageSmall`` und
  wird erst im Bildschritt aufgeloest.
* **Ob ein Bild ueberhaupt existiert.** ``Is Public Domain`` heisst
  gemeinfrei, nicht digitalisiert. Ein Teil der Auswahl faellt beim Download
  weg — die Stichprobe wird deshalb mit Reserve gezogen.
"""

from __future__ import annotations

import re
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# Spalten, die aus der 54-spaltigen CSV ueberhaupt gelesen werden. Alles andere
# kostet nur Speicher: die Datei ist ~317 MB.
COLUMNS = {
    "Object ID": "object_id",
    "Object Number": "object_number",
    "Is Public Domain": "is_public_domain",
    "Department": "department",
    "Object Name": "object_name",
    "Title": "title",
    "Culture": "culture",
    "Period": "period",
    "Artist Display Name": "artist",
    "Object Date": "object_date",
    "Object Begin Date": "begin_date",
    "Object End Date": "end_date",
    "Medium": "medium",
    "Classification": "classification",
    "Link Resource": "link",
}

# Facetten im Viewer. Bewusst vier — siehe Anti-Ziele im README.
#
# Die Auswahl ist gemessen, nicht geraten. Ueber die 248.472 gemeinfreien
# Objekte:
#
#   department       19 Werte, Top-8 deckt 83 %   -> Facette und Label
#   classification  693 Werte, Top-8 deckt 60 %   -> Facette (gekappt)
#   culture       5.242 Werte, Top-8 deckt 71 %   -> Facette (gekappt)
#   medium       37.876 Werte, Top-8 deckt 26 %   -> UNBRAUCHBAR
#
# `medium` ist zu fein ("Stone, probably jet; incised") und traegt als Filter
# nichts. `period` wird aus den numerischen Datumsfeldern zu Jahrhunderten
# gebuendelt und landet bei ~23 Werten.
FACETS = ("department", "classification", "culture", "period")

# Obergrenze fuer die Werteliste einer Facette. Alles darueber wird zu "Andere"
# zusammengefasst: eine Auswahlliste mit 5.000 Eintraegen ist kein Filter.
FACET_CAP = 16

# Die Auswahl wird auf die groessten Labelklassen beschraenkt, statt alle 19
# Abteilungen aufzunehmen und die kleinen spaeter zusammenzufassen.
#
# Der Unterschied ist nicht kosmetisch. Die Stichprobe ist nach Label
# geschichtet, macht die Klassen also ungefaehr gleich gross — eine
# anschliessende "die groessten sieben behalten ihre Farbe"-Regel waere danach
# willkuerlich und wuerde bei 19 Abteilungen zwoelf davon in einen einzigen
# Sammeltopf werfen, der dann zwei Drittel aller Punkte enthaelt. Ein grauer
# Klumpen mit 68 % der Wolke ist weder ein Farbkanal noch ein Testfall.
#
# Vorher zu begrenzen loest beides auf einmal: das Label wird zugleich der
# Farbkanal und die Wahrheitsgrundlage des Go/No-Go-Tors, mit acht
# ausgewogenen, visuell verschiedenen Klassen. Die acht groessten Abteilungen
# decken 83 % der gemeinfreien Sammlung ab.
MAX_LABEL_CLASSES = 8
OTHER = "Andere"


class CorpusError(RuntimeError):
    """Der Korpus laesst sich nicht bauen."""


# ---------------------------------------------------------------------------
# Herunterladen
# ---------------------------------------------------------------------------


def download_metadata(
    url: str,
    dest: Path,
    *,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
    chunk: int = 1 << 20,
) -> Path:
    """Laedt die Metadaten-CSV, wiederaufnehmbar.

    Die Datei ist ~317 MB, und der Server unterstuetzt Range-Requests. Ein
    Abbruch bei 90 % soll nicht bedeuten, dass von vorne begonnen wird.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        have = dest.stat().st_size
        total = _remote_size(url)
        if total is not None and have == total:
            return dest
        if total is None:
            return dest
    else:
        have = 0
        if force and dest.exists():
            dest.unlink()

    request = urllib.request.Request(url)
    if have:
        request.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(request, timeout=120) as response:
        # 206 = Fortsetzung akzeptiert, 200 = Server ignoriert den Range und
        # schickt alles; dann muss auch von vorne geschrieben werden.
        resuming = response.status == 206
        mode = "ab" if resuming and have else "wb"
        if not resuming:
            have = 0
        total = have + int(response.headers.get("Content-Length", 0))

        with dest.open(mode) as fh:
            while block := response.read(chunk):
                fh.write(block)
                have += len(block)
                if progress:
                    progress(have, total)

    return dest


def _remote_size(url: str) -> int | None:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lesen und filtern
# ---------------------------------------------------------------------------


def read_metadata(path: Path) -> pa.Table:
    """Liest die CSV und benennt die Spalten um.

    Alles als String einlesen: die Met-CSV hat leere Felder und gemischte
    Formate in den Datumsspalten, und ein Typfehler beim Parsen wuerde die
    gesamte Datei unlesbar machen statt nur ein Feld.
    """
    path = Path(path)
    if not path.is_file():
        raise CorpusError(
            f"{path} fehlt. Zuerst herunterladen: uv run --project pipeline lse fetch"
        )

    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=1 << 24),
        # Ohne `newlines_in_values` zerlegt pyarrow die Datei in Bloecke und
        # trifft dabei mitten in ein gequotetes Feld — die Met traegt
        # Zeilenumbrueche in Beschreibungs- und Massfeldern. Der Fehler
        # ("Expected 54 columns, got 22") zeigt dann auf eine harmlose Zeile
        # und verschweigt die eigentliche Ursache. Der Preis ist sequentielles
        # Parsen, also etwas laengere Laufzeit.
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        convert_options=pacsv.ConvertOptions(
            include_columns=list(COLUMNS),
            column_types={name: pa.string() for name in COLUMNS},
            strings_can_be_null=True,
        ),
    )
    return table.rename_columns([COLUMNS[name] for name in table.column_names])


def _clean(value: object) -> str:
    """Normalisiert ein Metadatenfeld zu einem gestutzten String."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value)).strip()
    # Die Met-CSV enthaelt Felder, die nur aus Leerzeichen bestehen.
    return "" if text in {"", "|"} else re.sub(r"\s+", " ", text)


def _first_segment(value: str) -> str:
    """Nimmt den ersten Teil eines mehrwertigen Feldes.

    Die Met traegt Mehrfachwerte pipe-getrennt ein ("Gold|Silver"). Fuer eine
    Facette ist der erste Wert die brauchbare Naeherung — eine Mehrfachauswahl
    waere ein anderes UI-Konzept und steht nicht im Plan.
    """
    return _clean(value.split("|")[0]) if value else ""


def bucket_period(begin: str, end: str, period: str) -> str:
    """Bildet eine Epoche auf ein grobes Jahrhundert ab.

    ``Period`` ist bei der Met sehr uneinheitlich befuellt (von "Edo period
    (1615-1868)" bis leer). Als Facette ist ein Jahrhundert aus den
    numerischen Datumsfeldern deutlich brauchbarer — und die numerischen Felder
    sind fast durchgaengig gesetzt.
    """
    for raw in (begin, end):
        try:
            year = int(float(raw))
        except (TypeError, ValueError):
            continue
        if year == 0:
            continue
        if year < -500:
            return "vor 500 v. Chr."
        if year < 0:
            return "500 v. Chr. – 0"
        century = (year // 100) + 1
        return f"{century}. Jh."
    return _first_segment(period) or "unbekannt"


def cap_facet(values: list[str], cap: int = FACET_CAP) -> tuple[list[str], list[str]]:
    """Behaelt die haeufigsten ``cap`` Werte und fasst den Rest zu "Andere".

    Rueckgabe: die gekappten Werte und das Vokabular in Haeufigkeitsreihenfolge.
    """
    from collections import Counter

    counts = Counter(values)
    keep = [name for name, _ in counts.most_common(cap) if name != OTHER]
    keep_set = set(keep)
    capped = [value if value in keep_set else OTHER for value in values]
    vocabulary = keep + ([OTHER] if any(v == OTHER for v in capped) else [])
    return capped, vocabulary


@dataclass
class CorpusStats:
    """Was der Filter weggeworfen hat — gehoert ins Manifest, nicht ins Log."""

    total: int
    public_domain: int
    with_title: int
    after_dedup: int
    selected: int
    label_classes: int
    dropped_classes: list[str]
    facet_sizes: dict[str, int]


def build_corpus(
    table: pa.Table,
    *,
    n: int,
    label_field: str = "department",
    seed: int = 0,
    min_class_size: int = 20,
    max_label_classes: int = MAX_LABEL_CLASSES,
    facet_cap: int = FACET_CAP,
) -> tuple[pa.Table, CorpusStats]:
    """Filtert, entdoppelt und zieht eine geschichtete Stichprobe.

    Geschichtet nach dem Label, nicht gleichverteilt: die Met-Abteilungen sind
    extrem unterschiedlich gross (Drawings and Prints stellt allein rund die
    Haelfte der Sammlung). Eine einfache Zufallsstichprobe waere zu drei
    Vierteln eine einzige Abteilung — und das Go/No-Go-Tor haette dann nichts
    zu trennen.
    """
    total = table.num_rows
    columns = {name: table.column(name).to_pylist() for name in table.column_names}

    rows: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    public_domain = 0
    with_title = 0

    for i in range(total):
        if _clean(columns["is_public_domain"][i]).lower() != "true":
            continue
        public_domain += 1

        title = _clean(columns["title"][i])
        if not title:
            continue
        with_title += 1

        # Entdopplung ueber den fachlichen Schluessel. Die Met fuehrt Serien und
        # Blattfolgen mit identischem Titel und Objektnamen; unbehandelt bilden
        # die ultradichte Mikrocluster, die die Ansicht dominieren und den Raum
        # "strukturiert" aussehen lassen, obwohl es nur Wiederholungen sind.
        key = "|".join(
            (
                title.casefold(),
                _clean(columns["artist"][i]).casefold(),
                _clean(columns["object_date"][i]).casefold(),
                _clean(columns["medium"][i]).casefold(),
            )
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)

        rows.append(
            {
                "object_id": _clean(columns["object_id"][i]),
                "object_number": _clean(columns["object_number"][i]),
                "title": title,
                "artist": _clean(columns["artist"][i]),
                "object_name": _first_segment(_clean(columns["object_name"][i])),
                "object_date": _clean(columns["object_date"][i]),
                "department": _clean(columns["department"][i]) or "unbekannt",
                "culture": _first_segment(_clean(columns["culture"][i])) or "unbekannt",
                "period": bucket_period(
                    _clean(columns["begin_date"][i]),
                    _clean(columns["end_date"][i]),
                    _clean(columns["period"][i]),
                ),
                "classification": _first_segment(_clean(columns["classification"][i]))
                or "unbekannt",
                # Kein Filter, aber im Detailpanel sichtbar — und zu fein
                # gegliedert fuer eine Facette (37.876 Werte).
                "medium": _first_segment(_clean(columns["medium"][i])) or "unbekannt",
                "link": _clean(columns["link"][i]),
            }
        )

    after_dedup = len(rows)
    if after_dedup == 0:
        raise CorpusError("Nach dem Filtern ist nichts uebrig — stimmt die CSV?")

    if label_field not in FACETS:
        raise CorpusError(f"label_field {label_field!r} ist keine der Facetten {FACETS}.")

    selected, dropped = _stratified_sample(
        rows, label_field, n, seed, min_class_size, max_label_classes
    )

    # Facetten erst nach der Stichprobe kappen: die Auswahlliste im Viewer soll
    # nur Werte enthalten, die auch vorkommen.
    #
    # Das Label ist ausgenommen. Es ist bereits durch `max_label_classes`
    # begrenzt, und es zusaetzlich zu kappen wuerde die Wahrheitsgrundlage des
    # Go/No-Go-Tors veraendern — man wuerde gegen eine Gruppierung validieren
    # statt gegen die echten Klassen.
    facet_sizes: dict[str, int] = {}
    for facet in FACETS:
        if facet == label_field:
            facet_sizes[facet] = len({row[facet] for row in selected})
            continue
        capped, vocabulary = cap_facet([row[facet] for row in selected], facet_cap)
        for row, value in zip(selected, capped):
            row[facet] = value
        facet_sizes[facet] = len(vocabulary)

    stats = CorpusStats(
        total=total,
        public_domain=public_domain,
        with_title=with_title,
        after_dedup=after_dedup,
        selected=len(selected),
        label_classes=len({row[label_field] for row in selected}),
        dropped_classes=dropped,
        facet_sizes=facet_sizes,
    )
    return pa.Table.from_pylist(selected), stats


def _stratified_sample(
    rows: list[dict[str, str]],
    label_field: str,
    n: int,
    seed: int,
    min_class_size: int,
    max_label_classes: int,
) -> tuple[list[dict[str, str]], list[str]]:
    """Zieht ungefaehr gleich viele Zeilen je Labelklasse.

    Zwei Beschraenkungen, in dieser Reihenfolge:

    1. Klassen unterhalb ``min_class_size`` fallen weg — eine Handvoll Punkte
       bildet im 3D-Raum keinen erkennbaren Cluster.
    2. Von den verbleibenden werden nur die ``max_label_classes`` groessten
       genommen. Das muss **vor** dem Ziehen passieren: danach sind alle
       Klassen gleich gross und "die groessten" waere eine Zufallsauswahl.

    Rueckgabe: die gezogenen Zeilen und die Namen der verworfenen Klassen.
    """
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        buckets.setdefault(row[label_field], []).append(row)

    buckets = {key: items for key, items in buckets.items() if len(items) >= min_class_size}
    if not buckets:
        raise CorpusError(
            f"Keine Klasse in {label_field!r} erreicht {min_class_size} Elemente."
        )

    ranked = sorted(buckets, key=lambda key: len(buckets[key]), reverse=True)
    kept = ranked[:max_label_classes]
    dropped = ranked[max_label_classes:]
    buckets = {key: buckets[key] for key in kept}

    rng = np.random.default_rng(seed)
    per_class = max(1, n // len(buckets))

    picked: list[dict[str, str]] = []
    leftovers: list[dict[str, str]] = []
    for key in sorted(buckets):
        items = buckets[key]
        order = rng.permutation(len(items))
        take = min(per_class, len(items))
        picked.extend(items[j] for j in order[:take])
        leftovers.extend(items[j] for j in order[take:])

    # Auffuellen aus den grossen Klassen, falls kleine Klassen das Kontingent
    # nicht ausschoepfen konnten.
    if len(picked) < n and leftovers:
        order = rng.permutation(len(leftovers))
        for j in order[: n - len(picked)]:
            picked.append(leftovers[j])

    # Stabile, aber vom Label unabhaengige Reihenfolge: sonst liegen alle
    # Objekte einer Abteilung zusammen, und jeder spaetere Teilbatch waere
    # systematisch verzerrt.
    order = rng.permutation(len(picked))
    return [picked[j] for j in order[:n]], dropped


def write_corpus(table: pa.Table, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def read_corpus(path: Path) -> pa.Table:
    path = Path(path)
    if not path.is_file():
        raise CorpusError(f"{path} fehlt — zuerst `lse fetch` ausfuehren.")
    return pq.read_table(path)


def label_encode(values: Iterable[str]) -> tuple[np.ndarray, list[str]]:
    """Bildet Labelstrings auf Indizes ab, alphabetisch stabil."""
    values = list(values)
    names = sorted(set(values))
    lookup = {name: index for index, name in enumerate(names)}
    return np.array([lookup[value] for value in values], dtype=np.int64), names
