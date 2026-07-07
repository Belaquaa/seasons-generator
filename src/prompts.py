"""Сборка матрицы промптов (сезон x тип) из prompts.yaml + config.yaml."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PromptItem:
    season: str
    ptype: str
    prompt: str
    negative_prompt: str | None


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: Path | None = None) -> dict:
    return _load_yaml(path or ROOT / "config.yaml")


def load_prompts(path: Path | None = None) -> dict:
    return _load_yaml(path or ROOT / "prompts" / "prompts.yaml")


def _fmt(template: str, ctx: dict) -> str:
    text = template.format(**ctx)
    return re.sub(r"\s+", " ", text).strip()


def build_prompt_matrix(config: dict, prompts: dict) -> list[PromptItem]:
    base = prompts["base"]
    seasons = prompts["seasons"]
    ptypes = prompts["prompt_types"]

    items: list[PromptItem] = []
    for season_key in config["seasons"]:
        season = seasons[season_key]
        ctx = {**base, **season}
        for ptype_key in config["prompt_types"]:
            spec = ptypes[ptype_key]
            prompt = _fmt(spec["template"], ctx)
            negative = None
            if spec.get("has_negative"):
                negative = _fmt(spec["negative_template"], ctx)
            items.append(PromptItem(season_key, ptype_key, prompt, negative))
    return items


def main() -> None:
    config = load_config()
    prompts = load_prompts()
    matrix = build_prompt_matrix(config, prompts)
    print(f"Матрица: {len(matrix)} промптов "
          f"({len(config['seasons'])} сезонов x {len(config['prompt_types'])} типов)\n")
    for it in matrix:
        print(f"[{it.season} / {it.ptype}]")
        print(f"  prompt:   {it.prompt}")
        if it.negative_prompt:
            print(f"  negative: {it.negative_prompt}")
        print()


if __name__ == "__main__":
    main()
