"""Bilder beschaffen: URLs aus der Objekt-API, dann herunterladen und verkleinern.

**Laeuft nicht in der Entwicklungs-Session** — ``collectionapi.metmuseum.org``
und ``images.metmuseum.org`` sind dort durch die Egress-Richtlinie gesperrt
(siehe docs/decisions.md E8). Der Code ist hier geschrieben und ausgefuehrt wird
er lokal.

Zwei Eigenschaften, die nicht optional sind:

* **Wiederaufnehmbar.** Ein Lauf ueber 100.000 Objekte stirbt irgendwann — an
  einem Timeout, einem korrupten JPEG, einem vollen Dateisystem. Ohne
  Fortschrittsjournal beginnt er danach von vorne.
* **Hoeflich.** Die Met stellt die Daten kostenlos bereit. Begrenzte
  Parallelitaet und ein Timeout sind das Mindeste; die Vorgaben hier sind
  bewusst zurueckhaltend.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

JOURNAL = "done.jsonl"

# Die Objekt-API liefert mehrere Bildgroessen. `primaryImageSmall` ist bereits
# auf die lange Kante ~800 px begrenzt und damit fuer ein 224er-Thumbnail
# reichlich — die Vollaufloesung zu ziehen waere ein Vielfaches an Daten fuer
# ein Bild, das anschliessend ohnehin verkleinert wird.
PREFERRED_FIELDS = ("primaryImageSmall", "primaryImage")


class ImageError(RuntimeError):
    """Bilder lassen sich nicht beschaffen."""


@dataclass
class ImageStats:
    requested: int
    resolved: int
    downloaded: int
    skipped: int
    failed: int
    seconds: float


class Journal:
    """Fortschrittsjournal, eine JSON-Zeile je erledigtem Objekt.

    Append-only und nach jeder Zeile geflusht: ein `kill -9` mitten im Lauf darf
    hoechstens den letzten Eintrag kosten, nicht die Datei.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        # Abgeschnittene letzte Zeile eines harten Abbruchs.
                        continue
                    self._done[str(entry["object_id"])] = entry
        self._handle = self.path.open("a", encoding="utf-8")

    def __contains__(self, object_id: object) -> bool:
        return str(object_id) in self._done

    def get(self, object_id: object) -> dict | None:
        return self._done.get(str(object_id))

    def record(self, entry: dict) -> None:
        self._done[str(entry["object_id"])] = entry
        self._handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    @property
    def done_count(self) -> int:
        return len(self._done)


def _client(timeout: float):
    import httpx

    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "latent-space-explorer/0.1 (Hobbyprojekt; CC0-Daten)"},
    )


def fetch_images(
    object_ids: Iterable[object],
    dest: Path,
    *,
    api: str,
    size: int = 224,
    workers: int = 8,
    timeout: float = 30.0,
    progress: Callable[[int, int], None] | None = None,
) -> ImageStats:
    """Loest Bild-URLs auf, laedt und verkleinert — wiederaufnehmbar.

    Legt ``dest/<object_id>.jpg`` an und fuehrt ``dest/done.jsonl``. Objekte
    ohne Bild werden ebenfalls journalisiert, damit sie beim naechsten Lauf
    nicht erneut angefragt werden: **`Is Public Domain` heisst gemeinfrei, nicht
    digitalisiert**, und ein spuerbarer Teil der Auswahl hat schlicht kein Bild.
    """
    from PIL import Image

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    journal = Journal(dest / JOURNAL)

    ids = [str(value) for value in object_ids]
    pending = [value for value in ids if value not in journal]
    started = time.perf_counter()
    downloaded = failed = resolved = 0

    def handle(object_id: str) -> dict:
        with _client(timeout) as client:
            response = client.get(f"{api}/{object_id}")
            response.raise_for_status()
            payload = response.json()

            url = ""
            for field in PREFERRED_FIELDS:
                candidate = (payload.get(field) or "").strip()
                if candidate:
                    url = candidate
                    break
            if not url:
                return {"object_id": object_id, "status": "no_image"}

            image_response = client.get(url)
            image_response.raise_for_status()
            raw = dest / f"{object_id}.orig"
            raw.write_bytes(image_response.content)

        try:
            with Image.open(raw) as image:
                image = image.convert("RGB")
                # Quadratisch zuschneiden, dann skalieren: ein verzerrtes Bild
                # wuerde dem Modell eine Geometrie zeigen, die es nie gesehen hat.
                short = min(image.size)
                left = (image.width - short) // 2
                top = (image.height - short) // 2
                image = image.crop((left, top, left + short, top + short))
                image = image.resize((size, size), Image.LANCZOS)
                image.save(dest / f"{object_id}.jpg", "JPEG", quality=90)
        finally:
            raw.unlink(missing_ok=True)

        return {"object_id": object_id, "status": "ok", "url": url}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(handle, value): value for value in pending}
        for done, future in enumerate(as_completed(futures), start=1):
            object_id = futures[future]
            try:
                entry = future.result()
            except Exception as exc:
                entry = {
                    "object_id": object_id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                }
            journal.record(entry)
            if entry["status"] == "ok":
                downloaded += 1
                resolved += 1
            elif entry["status"] == "no_image":
                resolved += 1
            else:
                failed += 1
            if progress:
                progress(done, len(pending))

    journal.close()
    return ImageStats(
        requested=len(ids),
        resolved=resolved,
        downloaded=downloaded,
        skipped=len(ids) - len(pending),
        failed=failed,
        seconds=time.perf_counter() - started,
    )


def available_images(dest: Path, object_ids: Iterable[object]) -> tuple[list[int], list[Path]]:
    """Welche Objekte der Auswahl haben tatsaechlich ein Bild auf der Platte.

    Rueckgabe: Zeilenindizes in der uebergebenen Reihenfolge und die Pfade.
    Beides gehoert zusammen — die Indizes sind es, ueber die der Korpus
    anschliessend auf die tatsaechlich einbettbaren Zeilen reduziert wird, und
    genau dort wird die ID-Invariante sonst gebrochen.
    """
    dest = Path(dest)
    indices: list[int] = []
    paths: list[Path] = []
    for row, object_id in enumerate(object_ids):
        path = dest / f"{object_id}.jpg"
        if path.is_file():
            indices.append(row)
            paths.append(path)
    return indices, paths
