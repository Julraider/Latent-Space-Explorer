"""Laden und Zugriff auf ``config.toml``.

Es gibt genau eine Konfigurationsdatei, sie liegt im Repo-Wurzelverzeichnis, und
kein Modul darf Pfade oder Parameter hart verdrahten. Jeder Artefaktlauf wird mit
der Config gestempelt, die ihn erzeugt hat — sonst ist "welcher Lauf hat diese
Koordinaten produziert?" nach wenigen Wochen unbeantwortbar.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "config.toml"

# Aktuelle Version des Datenvertrags. Wird gegen das ``schema_version`` in der
# Config und in jedem ``manifest.json`` geprueft.
SCHEMA_VERSION = 1


class ConfigError(RuntimeError):
    """Die Konfiguration fehlt, ist unlesbar oder in sich widerspruechlich."""


def find_repo_root(start: Path | None = None) -> Path:
    """Sucht ``config.toml`` aufwaerts vom Startpunkt.

    Damit funktionieren Aufrufe aus ``pipeline/``, aus dem Wurzelverzeichnis und
    aus einem Test-Arbeitsverzeichnis gleichermassen.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / CONFIG_FILENAME).is_file():
            return candidate
    raise ConfigError(
        f"{CONFIG_FILENAME} nicht gefunden (gesucht ab {here} aufwaerts). "
        "Aus dem Repo heraus aufrufen."
    )


@dataclass(frozen=True)
class Config:
    """Gelesene Konfiguration plus das Wurzelverzeichnis, aus dem sie stammt."""

    root: Path
    data: dict[str, Any]

    # -- Abschnitte ------------------------------------------------------

    @property
    def corpus(self) -> dict[str, Any]:
        return self.data["corpus"]

    @property
    def model(self) -> dict[str, Any]:
        return self.data["model"]

    @property
    def projection(self) -> dict[str, Any]:
        return self.data["projection"]

    @property
    def validation(self) -> dict[str, Any]:
        return self.data["validation"]

    @property
    def artifacts(self) -> dict[str, Any]:
        return self.data["artifacts"]

    @property
    def viewer(self) -> dict[str, Any]:
        return self.data["viewer"]

    # -- Abgeleitete Pfade -----------------------------------------------

    @property
    def runs_dir(self) -> Path:
        return self.root / self.artifacts["dir"]

    @property
    def sample_dir(self) -> Path:
        return self.root / self.artifacts["sample_dir"]

    def run_dir(self, name: str) -> Path:
        return self.runs_dir / name


def load(start: Path | None = None) -> Config:
    """Laedt und validiert die Konfiguration."""
    root = find_repo_root(start)
    path = root / CONFIG_FILENAME
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    found = data.get("schema_version")
    if found != SCHEMA_VERSION:
        raise ConfigError(
            f"{path}: schema_version {found!r}, erwartet {SCHEMA_VERSION}. "
            "Der Datenvertrag hat sich geaendert — siehe docs/decisions.md E7."
        )

    for section in ("corpus", "model", "projection", "validation", "artifacts", "viewer"):
        if section not in data:
            raise ConfigError(f"{path}: Abschnitt [{section}] fehlt.")

    _check_invariants(path, data)
    return Config(root=root, data=data)


def _check_invariants(path: Path, data: dict[str, Any]) -> None:
    """Prueft die Zusagen, die anderswo als gegeben vorausgesetzt werden."""
    proj = data["projection"]

    # densMAP und umap.transform() schliessen sich aus (umap_.py:3087). Der
    # Fallback-Pfad fuer den Live-Prompt haengt daran; siehe docs/decisions.md E4.
    if proj.get("umap_densmap"):
        raise ConfigError(
            f"{path}: projection.umap_densmap muss false bleiben. densMAP ist mit "
            "umap.transform() inkompatibel und wuerde den Live-Prompt-Fallback "
            "verbauen (siehe docs/decisions.md E4)."
        )

    # Auf L2-normalisierten Vektoren ist cosine monoton aequivalent zu euclidean;
    # cosine erzwingt zusaetzlich angular_rp_forest und ist langsamer. Beides zu
    # tun ist doppelte Arbeit ohne Gewinn.
    if proj.get("l2_normalize") and proj.get("umap_metric") == "cosine":
        raise ConfigError(
            f"{path}: bei l2_normalize=true ist umap_metric='euclidean' zu waehlen. "
            "Auf normalisierten Vektoren ist cosine monoton aequivalent, aber langsamer."
        )

    if proj.get("umap_n_components") != 3:
        raise ConfigError(
            f"{path}: projection.umap_n_components muss 3 sein — der Viewer liest "
            "drei Komponenten pro Punkt."
        )

    corpus = data["corpus"]
    label_field = corpus.get("label_field")
    if not label_field:
        raise ConfigError(
            f"{path}: corpus.label_field fehlt. Ohne bekanntes Label ist das "
            "Go/No-Go-Tor in Phase 0b wertlos (siehe docs/decisions.md E6)."
        )
