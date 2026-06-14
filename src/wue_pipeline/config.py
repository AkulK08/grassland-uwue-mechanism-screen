"""Configuration loading and path management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ProjectConfig:
    path: Path
    raw: Dict[str, Any]

    @property
    def project(self) -> Dict[str, Any]:
        return self.raw.get("project", {})

    @property
    def paths(self) -> Dict[str, str]:
        return self.raw.get("paths", {})

    @property
    def analysis(self) -> Dict[str, Any]:
        return self.raw.get("analysis", {})

    @property
    def products(self) -> Dict[str, List[str]]:
        return self.raw.get("products", {})

    @property
    def spatial(self) -> Dict[str, Any]:
        return self.raw.get("spatial", {})

    @property
    def catalog(self) -> Dict[str, Any]:
        return self.raw.get("data_catalog", {})

    @property
    def trait(self) -> Dict[str, Any]:
        return self.raw.get("trait_analysis", {})

    @property
    def mode(self) -> str:
        return str(self.project.get("mode", "production"))

    @property
    def seed(self) -> int:
        return int(self.project.get("random_seed", 42))

    def resolve(self, key: str) -> Path:
        p = Path(self.paths[key])
        if not p.is_absolute():
            p = self.path.parent.parent / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def file(self, key: str, name: str) -> Path:
        return self.resolve(key) / name


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = ProjectConfig(path=path, raw=raw)
    for k in ["raw", "demo", "interim", "processed", "external", "tables", "figures", "memos", "manuscript", "logs"]:
        if k in cfg.paths:
            cfg.resolve(k)
    return cfg
