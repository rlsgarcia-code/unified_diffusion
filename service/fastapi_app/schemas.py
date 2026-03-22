from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequestBody(BaseModel):
    model: str
    prompt: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 7.0
    seed: int | None = None
    device: str | None = None
    dtype: str | None = None
    output_path: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "sdxl.base",
                "prompt": "a dramatic product photo, studio lighting, clean background",
                "negative_prompt": "blurry, low quality",
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "guidance_scale": 6.5,
                "seed": 1234,
                "device": "mps",
                "dtype": "fp16",
                "output_path": "/Users/seu-user/Desktop/out.png",
            }
        }
    }


class VerifyFileRequestBody(BaseModel):
    path: str
    sha256: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "path": "/Users/seu-user/Downloads/model.safetensors",
                "sha256": "HASH_DO_CIVITAI",
            }
        }
    }


class RegisterLocalRequestBody(BaseModel):
    path: str
    registry_path: str = "custom-models.json"
    models_dir: str = "~/models/civitai"
    model_slug: str
    canonical_id: str
    provider: str = "diffusers"
    pipeline_type: str = "stable-diffusion-xl"
    default_revision: str = "main"
    license_hint: str = "__PREENCHER__"
    notes: str = "Local Civitai checkpoint."
    sha256: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "path": "/Users/seu-user/Downloads/model.safetensors",
                "registry_path": "/Users/seu-user/projetos/unified_diffusion/custom-models.json",
                "models_dir": "/Users/seu-user/models/civitai",
                "model_slug": "meu-modelo",
                "canonical_id": "local.civitai.meu-modelo",
                "provider": "diffusers",
                "pipeline_type": "stable-diffusion-xl",
                "default_revision": "main",
                "license_hint": "Verificar termos do criador.",
                "notes": "Checkpoint local registrado via API.",
                "sha256": "HASH_DO_CIVITAI",
            }
        }
    }


class GenerateResponseBody(BaseModel):
    out: str
    model: str
    provider: str
    cache_path: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "out": "/Users/seu-user/Desktop/out.png",
                "model": "sdxl.base@main",
                "provider": "diffusers",
                "cache_path": "/Users/seu-user/.cache/unified-diffusion/models/sdxl.base/main",
            }
        }
    }


class HealthResponseBody(BaseModel):
    status: str = Field(default="ok")

    model_config = {"json_schema_extra": {"example": {"status": "ok"}}}
