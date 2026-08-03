"""Tests fuer den Korpusaufbau.

Die Fixtures ahmen die Eigenheiten der echten Met-CSV nach, die beim Bauen
tatsaechlich aufgefallen sind: pipe-getrennte Mehrfachwerte, leere und
whitespace-only Felder, Duplikate durch Blattfolgen, und eine extrem
unausgewogene Klassenverteilung.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from lse import corpus


def _table(rows: list[dict[str, str]]) -> pa.Table:
    columns = list(corpus.COLUMNS.values())
    return pa.Table.from_pylist([{name: row.get(name, "") for name in columns} for row in rows])


def _row(**overrides) -> dict[str, str]:
    base = {
        "object_id": "1",
        "object_number": "1979.1",
        "is_public_domain": "True",
        "department": "Egyptian Art",
        "object_name": "Statue",
        "title": "Sitzende Figur",
        "culture": "Egyptian",
        "period": "New Kingdom",
        "artist": "",
        "object_date": "ca. 1350 BC",
        "begin_date": "-1350",
        "end_date": "-1300",
        "medium": "Limestone",
        "classification": "Sculpture",
        "link": "http://example.org/1",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Feldnormalisierung
# ---------------------------------------------------------------------------


def test_clean_handles_whitespace_and_none():
    assert corpus._clean(None) == ""
    assert corpus._clean("  ") == ""
    assert corpus._clean("|") == ""
    assert corpus._clean("  Gold   Leaf  ") == "Gold Leaf"


def test_first_segment_takes_leading_value():
    """Die Met traegt Mehrfachwerte pipe-getrennt ein."""
    assert corpus._first_segment("Gold|Silver|Enamel") == "Gold"
    assert corpus._first_segment("") == ""


@pytest.mark.parametrize(
    ("begin", "end", "expected"),
    [
        ("1850", "1899", "19. Jh."),
        ("-1350", "-1300", "vor 500 v. Chr."),
        ("-200", "-100", "500 v. Chr. – 0"),
        ("1", "50", "1. Jh."),
        ("", "", "unbekannt"),
        ("nicht-numerisch", "", "unbekannt"),
    ],
)
def test_bucket_period(begin, end, expected):
    assert corpus.bucket_period(begin, end, "") == expected


def test_bucket_period_falls_back_to_text():
    assert corpus.bucket_period("", "", "Edo period") == "Edo period"


def test_bucket_period_ignores_year_zero():
    """Die Met nutzt 0 als "unbekannt", nicht als Jahreszahl."""
    assert corpus.bucket_period("0", "1650", "") == "17. Jh."


# ---------------------------------------------------------------------------
# Filtern und Entdoppeln
# ---------------------------------------------------------------------------


def test_drops_non_public_domain():
    table = _table(
        [_row(object_id=str(i), is_public_domain="False") for i in range(30)]
        + [_row(object_id=str(100 + i), title=f"T{i}") for i in range(30)]
    )
    _, stats = corpus.build_corpus(table, n=10, min_class_size=5, max_label_classes=4)
    assert stats.public_domain == 30
    assert stats.after_dedup == 30


def test_drops_untitled():
    table = _table(
        [_row(object_id=str(i), title="") for i in range(20)]
        + [_row(object_id=str(100 + i), title=f"T{i}") for i in range(20)]
    )
    _, stats = corpus.build_corpus(table, n=10, min_class_size=5, max_label_classes=4)
    assert stats.public_domain == 40
    assert stats.with_title == 20


def test_dedup_removes_series_duplicates():
    """Blattfolgen mit identischem Titel bilden sonst ultradichte Mikrocluster.

    Der Raum sieht dann beeindruckend strukturiert aus und zeigt doch nur
    Wiederholungen.
    """
    duplicates = [_row(object_id=str(i), title="Tafel III") for i in range(25)]
    unique = [_row(object_id=str(100 + i), title=f"Einzelstueck {i}") for i in range(25)]
    _, stats = corpus.build_corpus(
        _table(duplicates + unique), n=30, min_class_size=5, max_label_classes=4
    )
    # 25 Duplikate schrumpfen auf eines.
    assert stats.after_dedup == 26


def test_dedup_keeps_same_title_different_artist():
    rows = [
        _row(object_id="1", title="Portrait", artist="A"),
        _row(object_id="2", title="Portrait", artist="B"),
    ] + [_row(object_id=str(10 + i), title=f"X{i}") for i in range(20)]
    _, stats = corpus.build_corpus(
        _table(rows), n=30, min_class_size=2, max_label_classes=4
    )
    assert stats.after_dedup == 22


# ---------------------------------------------------------------------------
# Stichprobe
# ---------------------------------------------------------------------------


def test_stratified_sample_balances_classes():
    """Ohne Schichtung waere die Stichprobe fast nur die groesste Abteilung."""
    rows = (
        [_row(object_id=str(i), department="Drawings and Prints", title=f"D{i}") for i in range(900)]
        + [_row(object_id=str(1000 + i), department="Egyptian Art", title=f"E{i}") for i in range(60)]
        + [_row(object_id=str(2000 + i), department="Asian Art", title=f"A{i}") for i in range(60)]
    )
    selected, stats = corpus.build_corpus(
        _table(rows), n=90, min_class_size=20, max_label_classes=8, seed=1
    )
    from collections import Counter

    counts = Counter(selected.column("department").to_pylist())
    assert stats.label_classes == 3
    # Ohne Schichtung waeren ~88 % "Drawings and Prints".
    assert max(counts.values()) - min(counts.values()) <= 1


def test_class_limit_applies_before_sampling():
    """Die Reihenfolge ist der eigentliche Punkt.

    Erst zu ziehen und dann die groessten Klassen zu behalten, waere sinnlos:
    nach der Schichtung sind alle Klassen gleich gross, "die groessten" waere
    eine Zufallsauswahl, und der Rest landet in einem Sammeltopf, der die
    Wolke dominiert.
    """
    rows = []
    for index, size in enumerate([500, 400, 300, 40, 30, 25]):
        name = f"Abteilung {index}"
        rows += [
            _row(object_id=f"{index}-{j}", department=name, title=f"{name} {j}")
            for j in range(size)
        ]

    selected, stats = corpus.build_corpus(
        _table(rows), n=120, min_class_size=20, max_label_classes=3, seed=2
    )
    kept = set(selected.column("department").to_pylist())

    assert kept == {"Abteilung 0", "Abteilung 1", "Abteilung 2"}
    assert stats.label_classes == 3
    assert len(stats.dropped_classes) == 3


def test_min_class_size_drops_tiny_classes():
    rows = [_row(object_id=str(i), department="Gross", title=f"G{i}") for i in range(60)]
    rows += [_row(object_id=f"k{i}", department="Winzig", title=f"K{i}") for i in range(3)]
    selected, _ = corpus.build_corpus(
        _table(rows), n=50, min_class_size=20, max_label_classes=8
    )
    assert "Winzig" not in set(selected.column("department").to_pylist())


def test_sample_order_is_label_independent():
    """Sonst waere jeder spaetere Teilbatch systematisch verzerrt."""
    rows = []
    for index in range(4):
        name = f"Abteilung {index}"
        rows += [
            _row(object_id=f"{index}-{j}", department=name, title=f"{name} {j}")
            for j in range(100)
        ]
    selected, _ = corpus.build_corpus(
        _table(rows), n=200, min_class_size=20, max_label_classes=4, seed=3
    )
    labels = selected.column("department").to_pylist()
    # Bei blockweiser Reihenfolge waeren fast alle Nachbarn gleich.
    changes = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    assert changes > len(labels) * 0.5


def test_sample_is_reproducible():
    rows = [
        _row(object_id=str(i), department=f"Abt {i % 4}", title=f"T{i}") for i in range(400)
    ]
    first, _ = corpus.build_corpus(_table(rows), n=50, min_class_size=20, max_label_classes=4, seed=7)
    second, _ = corpus.build_corpus(_table(rows), n=50, min_class_size=20, max_label_classes=4, seed=7)
    assert first.column("object_id").to_pylist() == second.column("object_id").to_pylist()


# ---------------------------------------------------------------------------
# Facetten
# ---------------------------------------------------------------------------


def test_cap_facet_collapses_long_tail():
    values = ["A"] * 50 + ["B"] * 40 + ["C"] * 30 + [f"selten-{i}" for i in range(200)]
    capped, vocabulary = corpus.cap_facet(values, cap=3)
    assert set(vocabulary) == {"A", "B", "C", corpus.OTHER}
    assert capped.count(corpus.OTHER) == 200


def test_label_field_is_never_capped():
    """Das Label ist die Wahrheitsgrundlage des Tors und darf nicht gruppiert werden."""
    rows = []
    for index in range(6):
        name = f"Abteilung {index}"
        rows += [
            _row(object_id=f"{index}-{j}", department=name, title=f"{name} {j}")
            for j in range(50)
        ]
    selected, stats = corpus.build_corpus(
        _table(rows), n=180, min_class_size=20, max_label_classes=6, facet_cap=2, seed=4
    )
    assert corpus.OTHER not in set(selected.column("department").to_pylist())
    assert stats.facet_sizes["department"] == 6


def test_facets_are_capped():
    rows = [
        _row(object_id=str(i), culture=f"Kultur-{i}", title=f"T{i}") for i in range(300)
    ]
    _, stats = corpus.build_corpus(
        _table(rows), n=200, min_class_size=20, max_label_classes=4, facet_cap=5
    )
    assert stats.facet_sizes["culture"] <= 6  # 5 plus "Andere"


# ---------------------------------------------------------------------------
# Kodierung und Ein-/Ausgabe
# ---------------------------------------------------------------------------


def test_label_encode_is_stable_and_alphabetical():
    labels, names = corpus.label_encode(["B", "A", "B", "C"])
    assert names == ["A", "B", "C"]
    assert labels.tolist() == [1, 0, 1, 2]


def test_write_read_roundtrip(tmp_path):
    rows = [_row(object_id=str(i), title=f"T{i}") for i in range(40)]
    selected, _ = corpus.build_corpus(_table(rows), n=20, min_class_size=5, max_label_classes=4)
    path = corpus.write_corpus(selected, tmp_path / "corpus.parquet")
    again = corpus.read_corpus(path)
    assert again.num_rows == selected.num_rows
    assert again.column("object_id").to_pylist() == selected.column("object_id").to_pylist()


def test_read_corpus_missing_is_clear(tmp_path):
    with pytest.raises(corpus.CorpusError, match="lse fetch"):
        corpus.read_corpus(tmp_path / "fehlt.parquet")


def test_empty_after_filtering_raises():
    table = _table([_row(object_id=str(i), is_public_domain="False") for i in range(10)])
    with pytest.raises(corpus.CorpusError):
        corpus.build_corpus(table, n=5, min_class_size=1, max_label_classes=4)
