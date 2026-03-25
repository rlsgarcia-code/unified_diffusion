from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequestBody(BaseModel):
    model: str = Field(description="Canonical model id, for example `sdxl.base`.")
    prompt: str = Field(description="Positive text prompt sent to the selected model.")
    negative_prompt: str | None = Field(
        default=None,
        description="Optional negative prompt used by providers that support it.",
    )
    width: int = Field(default=1024, description="Requested output width in pixels.")
    height: int = Field(default=1024, description="Requested output height in pixels.")
    steps: int = Field(default=30, description="Inference step count.")
    guidance_scale: float = Field(default=7.0, description="Classifier-free guidance scale.")
    seed: int | None = Field(
        default=None,
        description="Optional deterministic seed. When omitted the provider chooses one.",
    )
    device: str | None = Field(
        default=None,
        description="Optional device override such as `cpu`, `cuda`, or `mps`.",
    )
    dtype: str | None = Field(
        default=None,
        description="Optional dtype override such as `fp16`, `bf16`, or `fp32`.",
    )
    output_path: str | None = Field(
        default=None,
        description="Optional output image path. Defaults to `UDIFF_OUTPUT_DIR/out.png`.",
    )

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
    path: str = Field(description="Absolute or user-relative path to a local `.safetensors` file.")
    sha256: str | None = Field(
        default=None,
        description="Optional expected SHA-256 used to verify the local file before registration.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "path": "/Users/seu-user/Downloads/model.safetensors",
                "sha256": "HASH_DO_CIVITAI",
            }
        }
    }


class RegisterLocalRequestBody(BaseModel):
    path: str = Field(description="Path to the source `.safetensors` file to register.")
    registry_path: str = Field(
        default="custom-models.json",
        description="JSON registry file that will receive or update the custom model entry.",
    )
    models_dir: str = Field(
        default="~/models/civitai",
        description="Base directory where the normalized local model file will be moved.",
    )
    model_slug: str = Field(
        description="Stable folder slug used under `models_dir`, for example `omnigenx`.",
    )
    canonical_id: str = Field(
        description="Canonical model id that clients will later use in `/generate`.",
    )
    provider: str = Field(
        default="diffusers",
        description="Internal provider name. Use `diffusers` for standard local checkpoints.",
    )
    pipeline_type: str = Field(
        default="stable-diffusion-xl",
        description="Logical pipeline family such as `stable-diffusion-xl` or `flux`.",
    )
    default_revision: str = Field(
        default="main",
        description="Default logical revision stored in the registry entry.",
    )
    license_hint: str = Field(
        default="__PREENCHER__",
        description="Free-text license reminder copied into the registry entry.",
    )
    notes: str = Field(
        default="Local Civitai checkpoint.",
        description="Free-text notes copied into the registry entry.",
    )
    sha256: str | None = Field(
        default=None,
        description=(
            "Optional expected SHA-256. When supplied the API verifies it "
            "before moving the file."
        ),
    )

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
    out: str = Field(description="Filesystem path of the generated image.")
    model: str = Field(description="Resolved model id including revision.")
    provider: str = Field(description="Provider used to fulfill the generation request.")
    cache_path: str = Field(
        description="Deterministic cache location used for the model artifacts."
    )

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


class ModelsResponseBody(BaseModel):
    models: list[str] = Field(
        description="All canonical model ids available in the active registry."
    )

    model_config = {
        "json_schema_extra": {
            "example": {"models": ["sdxl.base", "sdxl.refiner", "local.civitai.omnigenx"]}
        }
    }


class PracticesResponseBody(BaseModel):
    practices: list[str] = Field(description="Operational recommendations for safe usage.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "practices": [
                    "Use `udiff verify-file --path ... --sha256 ...` before "
                    "registering a custom .safetensors file."
                ]
            }
        }
    }


class VerifyFileResponseBody(BaseModel):
    path: str
    file_name: str
    file_size_bytes: str
    sha256: str
    sha256_verified: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "path": "/Users/seu-user/Downloads/model.safetensors",
                "file_name": "model.safetensors",
                "file_size_bytes": "123456789",
                "sha256": "HASH_DO_CIVITAI",
                "sha256_verified": "true",
            }
        }
    }


class RegisterLocalResponseBody(BaseModel):
    canonical_id: str
    provider: str
    pipeline_type: str
    moved_to: str
    registry_path: str
    sample_command: str
    sha256_verified: str
    sha256_actual: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "canonical_id": "local.civitai.meu-modelo",
                "provider": "diffusers",
                "pipeline_type": "stable-diffusion-xl",
                "moved_to": "/Users/seu-user/models/civitai/meu-modelo/model.safetensors",
                "registry_path": "/Users/seu-user/projetos/unified_diffusion/custom-models.json",
                "sample_command": (
                    "UNIFIED_DIFFUSION_REGISTRY_PATH="
                    "/Users/seu-user/projetos/unified_diffusion/custom-models.json "
                    "uv run udiff run --model local.civitai.meu-modelo "
                    '--prompt "portrait photo of a person, studio lighting" --out meu-modelo.png'
                ),
                "sha256_verified": "true",
                "sha256_actual": "HASH_DO_CIVITAI",
            }
        }
    }
