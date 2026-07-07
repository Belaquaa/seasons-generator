"""Метрики сезонной генерации: структура (DINOv3 + SSIM) и сезон (CLIP)."""
from __future__ import annotations

import numpy as np

SEASON_TEXTS: dict[str, str] = {
    "summer": "an aerial photo of a residential area in summer with lush green "
              "vegetation and dry ground",
    "autumn": "an aerial photo of a residential area in autumn with yellow and "
              "orange foliage",
    "winter": "an aerial photo of a residential area in winter with light snow "
              "cover on the ground and rooftops",
    "deep_winter": "an aerial photo of a residential area in deep winter fully "
                   "covered by thick heavy snow",
}


def _to_pil(rgb: np.ndarray):
    from PIL import Image

    return Image.fromarray(rgb)


class Metrics:
    def __init__(self, eval_cfg: dict):
        self.cfg = eval_cfg
        self.device = eval_cfg.get("device", "cpu")
        self._dino = None
        self._dino_transform = None
        self._clip = None
        self._clip_processor = None
        self._season_text_emb: dict[str, "object"] = {}
        self._emb_cache: dict[tuple[str, str], "object"] = {}

    def _ensure_dino(self) -> None:
        if self._dino is not None:
            return
        import timm
        import torch

        name = self.cfg["structure_model"]
        model = timm.create_model(name, pretrained=True, num_classes=0)
        model.eval().to(self.device)
        data_cfg = timm.data.resolve_model_data_config(model)
        self._dino_transform = timm.data.create_transform(**data_cfg,
                                                          is_training=False)
        self._dino = model
        self._torch = torch

    def _dino_embed(self, rgb: np.ndarray, cache_key: str | None = None):
        if cache_key and ("dino", cache_key) in self._emb_cache:
            return self._emb_cache[("dino", cache_key)]
        self._ensure_dino()
        torch = self._torch
        x = self._dino_transform(_to_pil(rgb)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            emb = self._dino(x)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        if cache_key:
            self._emb_cache[("dino", cache_key)] = emb
        return emb

    def dino_similarity(self, src: np.ndarray, gen: np.ndarray,
                        src_key: str | None = None) -> float:
        a, b = self._dino_embed(src, src_key), self._dino_embed(gen)
        return float((a * b).sum().item())

    @staticmethod
    def ssim(src: np.ndarray, gen: np.ndarray) -> float:
        from skimage.color import rgb2gray
        from skimage.metrics import structural_similarity
        from skimage.transform import resize

        if src.shape != gen.shape:
            src = (resize(src, gen.shape, preserve_range=True)
                   .round().astype(np.uint8))
        return float(structural_similarity(rgb2gray(src), rgb2gray(gen),
                                           data_range=1.0))

    def _ensure_clip(self) -> None:
        if self._clip is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        name = self.cfg["season_model"]
        self._clip = CLIPModel.from_pretrained(name).eval().to(self.device)
        self._clip_processor = CLIPProcessor.from_pretrained(name)
        self._torch = torch
        with torch.inference_mode():
            for season, text in SEASON_TEXTS.items():
                inp = self._clip_processor(text=[text], return_tensors="pt",
                                           padding=True).to(self.device)
                self._season_text_emb[season] = self._clip_text_embed(inp)

    def _clip_text_embed(self, inp):
        # Явно text_model + text_projection: get_text_features в transformers v5
        # отдаёт output-объект, а pooler_output без проекции — не joint-пространство.
        torch = self._torch
        out = self._clip.text_model(input_ids=inp["input_ids"],
                                    attention_mask=inp.get("attention_mask"))
        emb = self._clip.text_projection(out.pooler_output)
        return torch.nn.functional.normalize(emb, dim=-1)

    def _clip_image_embed(self, rgb: np.ndarray, cache_key: str | None = None):
        if cache_key and ("clip", cache_key) in self._emb_cache:
            return self._emb_cache[("clip", cache_key)]
        self._ensure_clip()
        torch = self._torch
        inp = self._clip_processor(images=_to_pil(rgb),
                                   return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._clip.vision_model(pixel_values=inp["pixel_values"])
            emb = self._clip.visual_projection(out.pooler_output)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        if cache_key:
            self._emb_cache[("clip", cache_key)] = emb
        return emb

    def clip_season_prob(self, gen: np.ndarray, target_season: str) -> float:
        self._ensure_clip()
        if target_season not in SEASON_TEXTS:
            raise ValueError(f"Неизвестный сезон {target_season!r}, "
                             f"есть: {list(SEASON_TEXTS)}")
        torch = self._torch
        img = self._clip_image_embed(gen)
        seasons = list(SEASON_TEXTS)
        text = torch.cat([self._season_text_emb[s] for s in seasons], dim=0)
        logits = (img @ text.T).squeeze(0) * 100.0
        probs = torch.softmax(logits, dim=-1)
        return float(probs[seasons.index(target_season)].item())

    def clip_directional(self, src: np.ndarray, gen: np.ndarray,
                         target_season: str, source_season: str,
                         src_key: str | None = None) -> float:
        self._ensure_clip()
        torch = self._torch
        img_dir = (self._clip_image_embed(gen)
                   - self._clip_image_embed(src, src_key))
        txt_dir = (self._season_text_emb[target_season]
                   - self._season_text_emb[source_season])
        img_dir = torch.nn.functional.normalize(img_dir, dim=-1)
        txt_dir = torch.nn.functional.normalize(txt_dir, dim=-1)
        return float((img_dir * txt_dir).sum().item())

    def evaluate_pair(self, src: np.ndarray, gen: np.ndarray,
                      target_season: str, source_season: str,
                      src_key: str | None = None) -> dict:
        result = {
            "dino_sim": self.dino_similarity(src, gen, src_key),
            "ssim": self.ssim(src, gen),
            "clip_season_prob": self.clip_season_prob(gen, target_season),
        }
        if target_season != source_season:
            result["clip_directional"] = self.clip_directional(
                src, gen, target_season, source_season, src_key)
        else:
            result["clip_directional"] = None
        return result


def combined_score(row: dict, weights: dict, structure_threshold: float,
                   structure_weights: dict) -> dict:
    sw = structure_weights
    structure = float(sw["dino"] * row["dino_sim"] + sw["ssim"] * row["ssim"])
    season_parts = [row["clip_season_prob"]]
    cd = row.get("clip_directional")
    if cd is not None and not (isinstance(cd, float) and np.isnan(cd)):
        season_parts.append((cd + 1.0) / 2.0)
    season = float(np.mean(season_parts))
    disqualified = structure < structure_threshold
    score = 0.0 if disqualified else (weights["structure"] * structure
                                      + weights["season"] * season)
    return {
        "structure": round(structure, 4),
        "season": round(season, 4),
        "disqualified": disqualified,
        "score": round(score, 4),
    }
