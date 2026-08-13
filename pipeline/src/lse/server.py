"""Der optionale lokale Dienst fuer den Live-Prompt.

Bewusst klein und ohne Framework: die Standardbibliothek reicht fuer zwei
Endpunkte, und eine Abhaengigkeit, die nur ein Feature traegt, das viele
Besucher nie ausloesen, waere schlecht investiert.

**Die Architektur ist "statisch plus optionaler Dienst".** Rendern, Klicken,
Filtern und Stichwortsuche laufen vollstaendig clientseitig ueber die
Binaerdateien. Nur der Live-Prompt braucht das Modell. Das Frontend fragt beim
Laden ``/health``; antwortet niemand, bleibt das Prompt-Feld verborgen und alles
andere funktioniert. Andernfalls waere das ganze Projekt nur mit
``python main.py`` und einer RTX 4060 Ti benutzbar — und wuerde damit praktisch
niemand ansehen.

Endpunkte:

* ``GET  /health``  -> Zustand, Modell, Korpusgroesse
* ``POST /prompt``  -> ``{"text": "..."}`` -> Platzierung im Raum
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import artifacts, placement

MAX_BODY = 64 * 1024


def stub_encoder(dim: int) -> Callable[[str], np.ndarray]:
    """Deterministischer Ersatz fuer den Text-Encoder.

    Erzeugt aus dem Text einen reproduzierbaren Pseudo-Zufallsvektor. Damit
    laesst sich die **gesamte Kette** pruefen — HTTP, Platzierungsmathematik,
    Marker, Kamerafahrt — ohne Modellgewichte.

    Was er ausdruecklich NICHT liefert, ist Bedeutung: gleiche Prompts geben
    gleiche Punkte, aehnliche Prompts aber nicht aehnliche Punkte. Der
    Stub-Modus ist ein Integrationstest, kein Feature, und die Antwort sagt das
    auch (``"stub": true``).
    """

    def encode(text: str) -> np.ndarray:
        digest = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        return rng.normal(size=dim).astype(np.float32)

    return encode


class PromptService:
    """Haelt Embeddings, Koordinaten und den Encoder."""

    def __init__(
        self,
        run_dir: Path,
        *,
        encoder: Callable[[str], np.ndarray] | None = None,
        model_id: str = "stub",
        k: int = 12,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.model_id = model_id
        self.k = k
        self._lock = threading.Lock()

        manifest = artifacts.load_manifest(self.run_dir)
        self.manifest = manifest
        self.n = manifest.n

        stride = manifest.coords["coord_stride"]
        raw = artifacts.read_binary(
            self.run_dir / artifacts.COORDS_FILE, "<i2", (self.n, stride)
        )
        spec = artifacts.CoordSpec(
            offset=tuple(manifest.coords["offset"]),
            half=tuple(manifest.coords["half"]),
            min=tuple(manifest.coords["min"]),
            max=tuple(manifest.coords["max"]),
            radius=manifest.coords["radius"],
            source_centroid=tuple(manifest.coords["source_centroid"]),
            source_scale=manifest.coords["source_scale"],
            box=manifest.coords["box"],
        )
        self.coords = artifacts.dequantize_coords(raw, spec)
        self.radius = float(spec.radius)

        self.is_stub = encoder is None
        embeddings_path = self.run_dir / artifacts.EMBEDDINGS_FILE

        if embeddings_path.is_file():
            self.embeddings = placement.l2_normalize(np.load(embeddings_path))
            self.space = "embeddings"
        elif self.is_stub:
            # Ohne Embedding-Datei, aber im Stub-Modus: die 3D-Koordinaten
            # selbst als Suchraum nehmen.
            #
            # Das ist keine Notluesung, sondern die bessere Demo. Ein Prompt
            # landet dann bei seinen *raeumlichen* Nachbarn, die Platzierung
            # sieht plausibel aus und die Interaktion laesst sich vollstaendig
            # ausprobieren — ohne 3 MB Fantasievektoren im Repo. Bedeutung hat
            # sie weiterhin keine, und die Antwort sagt das.
            self.embeddings = placement.l2_normalize(self.coords.astype(np.float32))
            self.space = "coords"
        else:
            raise FileNotFoundError(
                f"{embeddings_path} fehlt. Mit echtem Modell braucht der Dienst die "
                "hochdimensionalen Embeddings — mit `lse project --keep-embeddings` "
                "erzeugen. Zum Ausprobieren ohne Modell: `lse serve --stub`."
            )

        self.dim = int(self.embeddings.shape[1])
        self.encode = encoder or stub_encoder(self.dim)

    def place(self, text: str) -> dict[str, Any]:
        # Der Encoder kann bei echten Modellen nicht threadsicher sein.
        with self._lock:
            vector = self.encode(text)
        result = placement.place(
            vector, self.embeddings, self.coords, k=self.k, radius=self.radius
        )
        payload = asdict(result)
        payload["text"] = text
        payload["stub"] = self.is_stub
        payload["model"] = self.model_id
        return payload

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.model_id,
            "stub": self.is_stub,
            "n": self.n,
            "dim": self.dim,
            "run": self.run_dir.name,
            "space": self.space,
            "corpus": self.manifest.corpus.get("id", "?"),
        }


def make_handler(service: PromptService, allow_origin: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # Der Viewer laeuft auf einem anderen Port als dieser Dienst; ohne
            # CORS-Freigabe blockt der Browser jede Antwort.
            self.send_header("Access-Control-Allow-Origin", allow_origin)
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(204, {})

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("/health", ""):
                self._send(200, service.health())
            else:
                self._send(404, {"error": "unbekannter Pfad"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/prompt":
                self._send(404, {"error": "unbekannter Pfad"})
                return

            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._send(413, {"error": "Anfrage zu gross"})
                return
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "kein gueltiges JSON"})
                return

            text = str(request.get("text") or "").strip()
            if not text:
                self._send(400, {"error": "leerer Prompt"})
                return

            try:
                self._send(200, service.place(text))
            except Exception as exc:  # pragma: no cover - defensiv
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *args) -> None:
            """Standardmaessig still — sonst rauscht jede Kamerabewegung mit."""

    return Handler


def serve(
    run_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    encoder: Callable[[str], np.ndarray] | None = None,
    model_id: str = "stub",
    k: int = 12,
    allow_origin: str = "*",
) -> ThreadingHTTPServer:
    """Baut den Server. Der Aufrufer entscheidet ueber ``serve_forever``."""
    service = PromptService(run_dir, encoder=encoder, model_id=model_id, k=k)
    server = ThreadingHTTPServer((host, port), make_handler(service, allow_origin))
    server.service = service
    return server
