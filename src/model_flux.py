"""Генераторы сезонных изображений за единым интерфейсом."""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GenRequest:
    source_name: str
    season: str
    ptype: str
    prompt: str
    negative_prompt: str | None
    seed: int


_SEASON_FX: dict[str, tuple[tuple[float, float, float], float]] = {
    "summer":      ((0.96, 1.10, 0.92), 0.00),
    "autumn":      ((1.18, 0.98, 0.72), 0.00),
    "winter":      ((1.00, 1.02, 1.10), 0.45),
    "deep_winter": ((1.00, 1.01, 1.06), 0.72),
}


class MockGenerator:

    name = "mock"
    model_id = "mock"

    def __init__(self, gcfg: dict | None = None):
        pass

    def generate(self, rgb: np.ndarray, req: GenRequest) -> np.ndarray:
        if req.season not in _SEASON_FX:
            raise ValueError(f"Неизвестный сезон {req.season!r}, есть: {list(_SEASON_FX)}")
        gains, white = _SEASON_FX[req.season]

        key = f"{req.ptype}|{req.season}|{req.source_name}|{req.seed}"
        h = zlib.crc32(key.encode("utf-8"))
        intensity = 0.85 + (h % 1000) / 1000.0 * 0.30

        img = rgb.astype(np.float64)
        for c in range(3):
            gain = 1.0 + (gains[c] - 1.0) * intensity
            img[:, :, c] *= gain
        w = min(white * intensity, 0.95)
        img = img * (1.0 - w) + 255.0 * w

        rng = np.random.default_rng(h)
        img += rng.normal(0.0, 2.0, size=img.shape)

        return np.clip(img, 0, 255).round().astype(np.uint8)


class FluxGenerator:
    """FLUX.1-Kontext-dev. По умолчанию bf16 (полная точность, весь пайплайн
    в VRAM). fp8 и gguf-q4 — опциональные режимы для GPU меньше 24/16 GB."""

    name = "flux"

    def __init__(self, gcfg: dict):
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                              "expandable_segments:True")
        import torch
        from diffusers import FluxKontextPipeline

        self._torch = torch
        self.model_id = gcfg["model_id"]
        mode = gcfg.get("dtype", "bf16")
        if mode == "gguf-q4":
            dtype = torch.float16
            from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
            from transformers import BitsAndBytesConfig, T5EncoderModel

            transformer = FluxTransformer2DModel.from_single_file(
                gcfg["gguf_url"],
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                config=gcfg["model_id"], subfolder="transformer",
                torch_dtype=dtype)
            transformer.to("cuda")  # T4-путь: держим в VRAM, RAM Colab всего 12.7GB
            text_encoder_2 = T5EncoderModel.from_pretrained(
                gcfg["model_id"], subfolder="text_encoder_2",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map={"": 0})
            self.pipe = FluxKontextPipeline.from_pretrained(
                gcfg["model_id"], transformer=transformer,
                text_encoder_2=text_encoder_2, torch_dtype=dtype)
            self.pipe.text_encoder.to("cuda")
            self.pipe.vae.to("cuda")
            self.pipe.vae.enable_tiling()
        else:
            dtype = torch.bfloat16
            self.pipe = FluxKontextPipeline.from_pretrained(
                gcfg["model_id"], torch_dtype=dtype
            )
            if mode == "fp8":
                from optimum.quanto import freeze, qfloat8, quantize

                quantize(self.pipe.transformer, weights=qfloat8)
                freeze(self.pipe.transformer)
        if mode != "gguf-q4":
            offload = gcfg.get("cpu_offload", "model")
            if offload == "model":
                self.pipe.enable_model_cpu_offload()
            elif offload == "sequential":
                self.pipe.enable_sequential_cpu_offload()
            else:
                self.pipe.to("cuda")
        self.steps = int(gcfg["num_inference_steps"])
        self.guidance = float(gcfg["guidance_scale"])
        self.true_cfg = float(gcfg.get("true_cfg_scale", 1.0))
        self._dtype = dtype
        self._emb: dict[str, tuple] = {}

    def prepare_prompts(self, prompts: list[str]) -> None:
        """Прекомпьют эмбеддингов промптов, затем выгрузка текст-энкодеров."""
        import gc

        torch = self._torch
        with torch.inference_mode():
            for text in dict.fromkeys(prompts):
                emb, pooled, _ = self.pipe.encode_prompt(
                    prompt=text, prompt_2=None, device="cuda",
                    num_images_per_prompt=1)
                # каст к dtype пайплайна, иначе dtype-конфликт (mat1/mat2, VAE)
                self._emb[text] = (emb.to("cpu", self._dtype),
                                   pooled.to("cpu", self._dtype))
        self.pipe.text_encoder_2 = None
        self.pipe.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()

    def generate(self, rgb: np.ndarray, req: GenRequest) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        emb, pooled = self._emb[req.prompt]
        kwargs = dict(
            image=Image.fromarray(rgb),
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=torch.Generator().manual_seed(req.seed),
            prompt_embeds=emb.to("cuda"),
            pooled_prompt_embeds=pooled.to("cuda"),
        )
        if req.negative_prompt:
            nemb, npooled = self._emb[req.negative_prompt]
            kwargs["negative_prompt_embeds"] = nemb.to("cuda")
            kwargs["negative_pooled_prompt_embeds"] = npooled.to("cuda")
            kwargs["true_cfg_scale"] = self.true_cfg
        out = self.pipe(**kwargs).images[0]
        return np.asarray(out.convert("RGB"), dtype=np.uint8)


class Flux2Generator:
    """Трансформер 64GB + энкодер Mistral 48GB не влезают в 95GB одновременно —
    работа через model_cpu_offload."""

    name = "flux2"
    model_id = "black-forest-labs/FLUX.2-dev"

    def __init__(self, gcfg: dict):
        import torch
        from diffusers import Flux2Pipeline

        self._torch = torch
        self.pipe = Flux2Pipeline.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()
        self.steps = int(gcfg["num_inference_steps"])
        self.guidance = float(gcfg.get("flux2_guidance_scale", 2.5))

    def generate(self, rgb: np.ndarray, req: GenRequest) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        out = self.pipe(
            prompt=req.prompt,
            image=[Image.fromarray(rgb)],
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=torch.Generator().manual_seed(req.seed),
        ).images[0]
        return np.asarray(out.convert("RGB"), dtype=np.uint8)


class SDXLImg2ImgGenerator:
    """CLIP режет промпт до 77 токенов."""

    name = "sdxl"
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, gcfg: dict):
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline

        self._torch = torch
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")
        self.pipe.vae.enable_tiling()
        self.strength = float(gcfg.get("sd_strength", 0.45))
        self.steps = int(gcfg["num_inference_steps"])
        self.guidance = float(gcfg.get("sd_guidance_scale", 7.5))

    def generate(self, rgb: np.ndarray, req: GenRequest) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        out = self.pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=Image.fromarray(rgb),
            strength=self.strength,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=torch.Generator().manual_seed(req.seed),
        ).images[0]
        return np.asarray(out.convert("RGB"), dtype=np.uint8)


class InstructPix2PixGenerator:

    name = "ip2p"
    model_id = "timbrooks/instruct-pix2pix"

    def __init__(self, gcfg: dict):
        import torch
        from diffusers import StableDiffusionInstructPix2PixPipeline

        self._torch = torch
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float16, safety_checker=None)
        self.pipe.to("cuda")
        self.steps = int(gcfg["num_inference_steps"])
        self.guidance = float(gcfg.get("sd_guidance_scale", 7.5))
        self.image_guidance = float(gcfg.get("image_guidance_scale", 1.5))

    def generate(self, rgb: np.ndarray, req: GenRequest) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        out = self.pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=Image.fromarray(rgb),
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            image_guidance_scale=self.image_guidance,
            generator=torch.Generator().manual_seed(req.seed),
        ).images[0]
        return np.asarray(out.convert("RGB"), dtype=np.uint8)


_BACKENDS = {
    "mock": MockGenerator,
    "flux": FluxGenerator,
    "flux2": Flux2Generator,
    "sdxl": SDXLImg2ImgGenerator,
    "ip2p": InstructPix2PixGenerator,
}


def get_generator(config: dict):
    gcfg = config["generation"]
    backend = gcfg.get("backend", "mock")
    if backend not in _BACKENDS:
        raise ValueError(f"Неизвестный backend {backend!r}, "
                         f"есть: {' | '.join(_BACKENDS)}")
    return _BACKENDS[backend](gcfg)
