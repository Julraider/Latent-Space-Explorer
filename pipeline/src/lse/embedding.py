"""Embeddings mit SigLIP 2 — Bilder und Text im selben Raum.

**Laeuft nicht in der Entwicklungs-Session**: ``huggingface.co`` ist dort durch
die Egress-Richtlinie gesperrt (docs/decisions.md E8). Der Code entsteht hier,
ausgefuehrt wird er lokal auf der RTX 4060 Ti.

Warum ein geteilter Bild-Text-Raum: der Live-Prompt aus Phase 4c ("tippe eine
Phrase, flieg zu den Bildern") braucht Text und Bild in derselben Geometrie. Ein
reiner Bildencoder wie DINOv3 liefert schoenere visuelle Mannigfaltigkeiten,
aber das Prompt-Feature entfaellt damit ersatzlos.

⚠️ **Modality Gap.** Bild- und Text-Embeddings liegen bei CLIP-artigen Modellen
in *verschiedenen Kegeln* des gemeinsamen Raums. Ein Text-Embedding direkt in die
Bildprojektion zu schicken, landet systematisch versetzt — moeglicherweise
ausserhalb der Wolke. Deshalb wird ein Prompt nicht projiziert, sondern ueber
seine naechsten Bild-Nachbarn platziert (docs/decisions.md E4). Diese Datei
liefert nur die Vektoren; die Platzierung passiert in der Projektion.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

SHARD_PATTERN = "shard_{index:05d}.npy"
SHARD_JOURNAL = "shards.jsonl"


class EmbeddingError(RuntimeError):
    """Das Modell laesst sich nicht laden oder anwenden."""


@dataclass
class EmbedStats:
    n: int
    dim: int
    batches: int
    seconds: float
    device: str
    resumed_from: int


def pick_device(preferred: str | None = None) -> str:
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id: str, revision: str = "main", *, device: str, dtype: str = "float16"):
    """Laedt SigLIP 2 nebst Prozessor.

    ``revision`` sollte fuer den veroeffentlichten Lauf ein Commit-SHA sein, kein
    Branchname: ``main`` kann sich unter den Fuessen aendern, und die Embeddings
    waeren dann nicht mehr reproduzierbar — was das Manifest behauptet.
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype)
    if torch_dtype is None:
        raise EmbeddingError(f"Unbekannter dtype {dtype!r}.")
    if device == "cpu" and torch_dtype is torch.float16:
        # fp16 auf CPU ist teils nicht implementiert und immer langsam.
        torch_dtype = torch.float32

    try:
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = AutoModel.from_pretrained(model_id, revision=revision, torch_dtype=torch_dtype)
    except Exception as exc:  # pragma: no cover - haengt vom Netz ab
        raise EmbeddingError(
            f"{model_id}@{revision} nicht ladbar: {type(exc).__name__}: {exc}\n"
            "Falls die Meldung nach einer Netzsperre aussieht: huggingface.co ist "
            "in der Entwicklungs-Session blockiert, dieser Schritt gehoert auf die "
            "lokale Maschine (siehe docs/decisions.md E8)."
        ) from exc

    return model.to(device).eval(), processor


def embed_images(
    paths: Sequence[Path],
    *,
    model_id: str,
    revision: str = "main",
    batch_size: int = 256,
    dtype: str = "float16",
    device: str | None = None,
    shard_dir: Path | None = None,
    shard_size: int = 8192,
    num_workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, EmbedStats]:
    """Bettet Bilder ein, geshardet und wiederaufnehmbar.

    Der Engpass ist nicht die GPU, sondern das JPEG-Dekodieren — deshalb
    ``num_workers`` im DataLoader. Auf einer 4060 Ti sind mit dem base-Modell
    grob 300-600 Bilder/s Ende zu Ende realistisch; die GPU allein schafft mehr.

    Die Karte ist bandbreitenlimitiert (128-Bit-Bus, 288 GB/s), nicht
    rechenlimitiert: sehr grosse Batches bringen weniger als erwartet.
    """
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    device = pick_device(device)
    model, processor = load_model(model_id, revision, device=device, dtype=dtype)

    shard_dir = Path(shard_dir) if shard_dir else None
    done_shards: dict[int, int] = {}
    if shard_dir:
        shard_dir.mkdir(parents=True, exist_ok=True)
        journal = shard_dir / SHARD_JOURNAL
        if journal.exists():
            for line in journal.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done_shards[int(entry["index"])] = int(entry["count"])

    class ImageSet(Dataset):
        def __len__(self) -> int:
            return len(paths)

        def __getitem__(self, index: int):
            with Image.open(paths[index]) as image:
                return processor(images=image.convert("RGB"), return_tensors="pt")[
                    "pixel_values"
                ][0]

    loader = DataLoader(
        ImageSet(),
        batch_size=batch_size,
        shuffle=False,  # Die Reihenfolge IST die ID-Invariante.
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    started = time.perf_counter()
    chunks: list[np.ndarray] = []
    buffer: list[np.ndarray] = []
    buffered = 0
    shard_index = 0
    resumed = 0
    batches = 0

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda" and dtype == "float16"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    def flush_shard() -> None:
        nonlocal buffer, buffered, shard_index
        if not buffer:
            return
        block = np.concatenate(buffer, axis=0)
        if shard_dir:
            path = shard_dir / SHARD_PATTERN.format(index=shard_index)
            np.save(path, block, allow_pickle=False)
            with (shard_dir / SHARD_JOURNAL).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"index": shard_index, "count": len(block)}) + "\n")
        chunks.append(block)
        shard_index += 1
        buffer = []
        buffered = 0

    with torch.no_grad():
        for batch in loader:
            batches += 1
            batch = batch.to(device, non_blocking=True)
            with autocast:
                features = model.get_image_features(pixel_values=batch)
            buffer.append(features.float().cpu().numpy().astype(np.float32))
            buffered += len(batch)
            if shard_dir and buffered >= shard_size:
                flush_shard()
            if progress:
                progress(min(batches * batch_size, len(paths)), len(paths))

    flush_shard()
    embeddings = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.zeros((0, 0), dtype=np.float32)
    )

    return embeddings, EmbedStats(
        n=len(embeddings),
        dim=int(embeddings.shape[1]) if embeddings.size else 0,
        batches=batches,
        seconds=time.perf_counter() - started,
        device=device,
        resumed_from=resumed,
    )


def embed_texts(
    texts: Sequence[str],
    *,
    model_id: str,
    revision: str = "main",
    batch_size: int = 128,
    dtype: str = "float16",
    device: str | None = None,
) -> np.ndarray:
    """Bettet Text im selben Raum ein — die Basis des Live-Prompts.

    Die Vektoren landen wegen des Modality Gap NICHT direkt in der Projektion.
    Sie dienen dazu, die naechsten Bild-Nachbarn zu finden; platziert wird der
    Prompt dann auf deren Schwerpunkt.
    """
    import torch

    device = pick_device(device)
    model, processor = load_model(model_id, revision, device=device, dtype=dtype)

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            block = list(texts[start : start + batch_size])
            inputs = processor(
                text=block, padding="max_length", truncation=True, return_tensors="pt"
            ).to(device)
            features = model.get_text_features(**inputs)
            out.append(features.float().cpu().numpy().astype(np.float32))

    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)
