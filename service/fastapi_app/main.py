from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from service.fastapi_app.schemas import (
    GenerateRequestBody,
    GenerateResponseBody,
    HealthResponseBody,
    RegisterLocalRequestBody,
    VerifyFileRequestBody,
)
from unified_diffusion import Diffusion, GenerateRequest, ModelNotFoundError, ProviderError
from unified_diffusion.operations import (
    best_usage_practices,
    configure_registry_path,
    register_local_model_entry,
    verify_local_file,
)
from unified_diffusion.settings import get_settings
from unified_diffusion.telemetry import log_event

app = FastAPI(
    title="unified_diffusion API",
    version="0.1.0",
    description=(
        "Thin HTTP layer over unified_diffusion. "
        "Use /docs for interactive examples and /practices for operational guidance."
    ),
)

BAD_REQUEST_RESPONSE = {
    400: {
        "description": "Invalid request or runtime validation failure",
        "content": {
            "application/json": {
                "example": {
                    "detail": (
                        "SHA-256 mismatch for 'model.safetensors': "
                        "expected HASH, got OTHER_HASH"
                    )
                }
            }
        },
    }
}


def _engine() -> Diffusion:
    configure_registry_path()
    settings = get_settings()
    return Diffusion(cache_dir=settings.cache_dir)


@app.get("/health", response_model=HealthResponseBody, summary="Health check")
def health() -> HealthResponseBody:
    return HealthResponseBody()


@app.get("/models", summary="List available model ids")
def models() -> dict[str, list[str]]:
    payload = {"models": _engine().list_models()}
    log_event("api.models", count=len(payload["models"]))
    return payload


@app.get("/practices", summary="Show best usage practices")
def practices() -> dict[str, list[str]]:
    payload = {"practices": best_usage_practices()}
    log_event("api.practices", count=len(payload["practices"]))
    return payload


@app.post(
    "/verify-file",
    summary="Verify a local safetensors file",
    responses=BAD_REQUEST_RESPONSE,
)
def verify_file(request: VerifyFileRequestBody) -> dict[str, str]:
    try:
        return verify_local_file(
            source_path=Path(request.path).expanduser(),
            expected_sha256=request.sha256,
        )
    except (FileNotFoundError, ValueError) as exc:
        log_event("api.verify_file_failed", path=request.path, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/register-local",
    summary="Register and move a local safetensors file",
    responses=BAD_REQUEST_RESPONSE,
)
def register_local(request: RegisterLocalRequestBody) -> dict[str, str]:
    try:
        return register_local_model_entry(
            source_path=Path(request.path).expanduser(),
            registry_path=Path(request.registry_path).expanduser(),
            models_dir=Path(request.models_dir).expanduser(),
            model_slug=request.model_slug,
            canonical_id=request.canonical_id,
            provider=request.provider,
            pipeline_type=request.pipeline_type,
            default_revision=request.default_revision,
            license_hint=request.license_hint,
            notes=request.notes,
            expected_sha256=request.sha256,
        )
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        log_event("api.register_local_failed", path=request.path, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/generate",
    response_model=GenerateResponseBody,
    summary="Generate an image",
    responses={
        400: {
            "description": "Model resolution, provider, or generation failure",
            "content": {
                "application/json": {
                    "example": {
                        "detail": (
                            "Unknown model 'local.missing'. "
                            "Use Diffusion.list_models() to inspect supported ids."
                        )
                    }
                }
            },
        }
    },
)
def generate(request: GenerateRequestBody) -> GenerateResponseBody:
    engine = _engine()
    output_dir = get_settings().output_dir
    output_path = (
        Path(request.output_path).expanduser()
        if request.output_path
        else output_dir / "out.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = engine.run(
            GenerateRequest(
                model=request.model,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                device=request.device,
                dtype=request.dtype,
            )
        )
    except (ModelNotFoundError, ProviderError, ValueError) as exc:
        log_event("api.generate_failed", model=request.model, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result.images[0].save(output_path)
    payload = GenerateResponseBody(
        out=str(output_path),
        model=result.model_resolved,
        provider=result.provider_used,
        cache_path=result.cache_path,
    )
    log_event("api.generate_completed", model=payload.model, out=payload.out)
    return payload
