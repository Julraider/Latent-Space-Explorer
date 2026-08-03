"""Tests fuer das Laden der Konfiguration und ihre Invarianten.

Die geprueften Invarianten sind keine Stilfragen: jede einzelne wuerde sonst
erst Phasen spaeter auffallen, wenn die Artefakte schon erzeugt sind.
"""

from __future__ import annotations

import tomllib

import pytest

from lse import config as config_module


def test_loads_repo_config():
    cfg = config_module.load()
    assert (cfg.root / "config.toml").is_file()
    assert cfg.data["schema_version"] == config_module.SCHEMA_VERSION
    assert cfg.corpus["id"]
    assert cfg.model["dim"] > 0


def test_finds_root_from_subdirectory():
    cfg = config_module.load()
    from_sub = config_module.load(cfg.root / "pipeline" / "src" / "lse")
    assert from_sub.root == cfg.root


def test_missing_config_is_reported_clearly(tmp_path):
    with pytest.raises(config_module.ConfigError, match="nicht gefunden"):
        config_module.find_repo_root(tmp_path)


def _write(tmp_path, data: dict) -> None:
    import json

    lines = ["schema_version = 1"]
    for section, values in data.items():
        lines.append(f"\n[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {json.dumps(value)}")
    (tmp_path / "config.toml").write_text("\n".join(lines), encoding="utf-8")


def _base() -> dict:
    return {
        "corpus": {"id": "x", "target_n": 10, "label_field": "department"},
        "model": {"id": "m", "revision": "main", "dim": 8},
        "projection": {
            "l2_normalize": True,
            "pca_components": 4,
            "umap_n_components": 3,
            "umap_n_neighbors": 15,
            "umap_min_dist": 0.05,
            "umap_metric": "euclidean",
            "umap_densmap": False,
            "random_state": 1,
        },
        "validation": {"knn_k": 15},
        "artifacts": {"dir": "data/runs", "sample_dir": "data/sample", "coord_box": 50.0},
        "viewer": {"max_point_px": 48.0},
    }


def test_rejects_densmap(tmp_path):
    """densMAP und umap.transform() schliessen sich aus (umap_.py:3087)."""
    data = _base()
    data["projection"]["umap_densmap"] = True
    _write(tmp_path, data)

    with pytest.raises(config_module.ConfigError, match="densmap"):
        config_module.load(tmp_path)


def test_rejects_cosine_with_l2_normalize(tmp_path):
    """Auf normalisierten Vektoren ist cosine aequivalent, aber langsamer."""
    data = _base()
    data["projection"]["umap_metric"] = "cosine"
    _write(tmp_path, data)

    with pytest.raises(config_module.ConfigError, match="euclidean"):
        config_module.load(tmp_path)


def test_rejects_non_3d_projection(tmp_path):
    data = _base()
    data["projection"]["umap_n_components"] = 2
    _write(tmp_path, data)

    with pytest.raises(config_module.ConfigError, match="muss 3 sein"):
        config_module.load(tmp_path)


def test_rejects_missing_label_field(tmp_path):
    """Ohne bekanntes Label ist das Go/No-Go-Tor in Phase 0b wertlos."""
    data = _base()
    del data["corpus"]["label_field"]
    _write(tmp_path, data)

    with pytest.raises(config_module.ConfigError, match="label_field"):
        config_module.load(tmp_path)


def test_rejects_wrong_schema_version(tmp_path):
    (tmp_path / "config.toml").write_text("schema_version = 99\n", encoding="utf-8")
    with pytest.raises(config_module.ConfigError, match="schema_version"):
        config_module.load(tmp_path)


def test_rejects_missing_section(tmp_path):
    data = _base()
    del data["viewer"]
    _write(tmp_path, data)

    with pytest.raises(config_module.ConfigError, match=r"\[viewer\]"):
        config_module.load(tmp_path)


def test_repo_config_passes_its_own_invariants():
    """Die echte config.toml muss die Regeln erfuellen, die sie aufstellt."""
    cfg = config_module.load()
    assert cfg.projection["umap_densmap"] is False
    assert cfg.projection["umap_n_components"] == 3
    if cfg.projection["l2_normalize"]:
        assert cfg.projection["umap_metric"] == "euclidean"


def test_config_is_valid_toml():
    cfg = config_module.load()
    with (cfg.root / "config.toml").open("rb") as fh:
        tomllib.load(fh)
