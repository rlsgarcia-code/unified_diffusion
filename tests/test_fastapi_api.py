import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from service.fastapi_app.main import app
from service.fastapi_app.schemas import DEFAULT_GENERATE_REQUEST_EXAMPLE
from unified_diffusion import GenerateResult


class FakeDiffusion:
    last_request = None

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir

    def list_models(self) -> list[str]:
        return ["sdxl.base", "local.civitai.omnigenx"]

    def run(self, request):
        type(self).last_request = request
        image = Image.new("RGB", (request.width, request.height), color="white")
        return GenerateResult(
            images=[image],
            seed_used=request.seed,
            model_resolved=f"{request.model}@main",
            provider_used="diffusers",
            cache_path="/tmp/model",
            metadata={},
        )


client = TestClient(app)


def test_root_redirects_to_docs() -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/docs"


def test_docs_page_is_available() -> None:
    response = client.get("/docs")

    assert response.status_code == 200
    assert "Swagger UI" in response.text


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models(monkeypatch) -> None:
    monkeypatch.setattr("service.fastapi_app.main.Diffusion", FakeDiffusion)

    response = client.get("/models")

    assert response.status_code == 200
    assert response.json()["models"] == ["sdxl.base", "local.civitai.omnigenx"]


def test_practices() -> None:
    response = client.get("/practices")

    assert response.status_code == 200
    assert response.json()["practices"]


def test_openapi_exposes_examples() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    payload = response.json()
    schemas = payload["components"]["schemas"]
    assert "example" in schemas["GenerateRequestBody"]
    assert "example" in schemas["VerifyFileRequestBody"]
    assert "example" in schemas["RegisterLocalRequestBody"]
    assert "example" in schemas["RegisterLocalResponseBody"]


def test_openapi_exposes_error_examples() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    payload = response.json()
    generate_post = payload["paths"]["/generate"]["post"]
    assert "400" in generate_post["responses"]
    assert "example" in generate_post["responses"]["400"]["content"]["application/json"]


def test_openapi_lists_register_local_in_docs() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    payload = response.json()
    register_post = payload["paths"]["/register-local"]["post"]
    assert register_post["summary"] == "Register and move a local safetensors file"
    assert register_post["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/RegisterLocalRequestBody"
    )
    response_schema_ref = register_post["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    assert response_schema_ref.endswith("/RegisterLocalResponseBody")


def test_verify_file(tmp_path: Path) -> None:
    source_path = tmp_path / "verify.safetensors"
    source_path.write_bytes(b"abc123")
    sha256 = hashlib.sha256(b"abc123").hexdigest()

    response = client.post("/verify-file", json={"path": str(source_path), "sha256": sha256})

    assert response.status_code == 200
    assert response.json()["sha256_verified"] == "true"


def test_register_local(tmp_path: Path) -> None:
    source_path = tmp_path / "demo.safetensors"
    source_path.write_bytes(b"abc123")
    registry_path = tmp_path / "custom-models.json"
    models_dir = tmp_path / "models"
    sha256 = hashlib.sha256(b"abc123").hexdigest()

    response = client.post(
        "/register-local",
        json={
            "path": str(source_path),
            "registry_path": str(registry_path),
            "models_dir": str(models_dir),
            "model_slug": "demo-model",
            "canonical_id": "local.demo-model",
            "sha256": sha256,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["canonical_id"] == "local.demo-model"
    assert (models_dir / "demo-model" / "model.safetensors").exists()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["local.demo-model"]["source"].endswith("demo-model/model.safetensors")


def test_generate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("service.fastapi_app.main.Diffusion", FakeDiffusion)
    monkeypatch.setenv("UDIFF_OUTPUT_DIR", str(tmp_path / "outputs"))

    response = client.post(
        "/generate",
        json={
            "model": "sdxl.base",
            "prompt": "hello",
            "output_path": str(tmp_path / "outputs" / "out.png"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "sdxl.base@main"
    assert Path(payload["out"]).exists()


def test_generate_uses_default_payload_for_empty_json(monkeypatch) -> None:
    monkeypatch.setattr("service.fastapi_app.main.Diffusion", FakeDiffusion)
    monkeypatch.setattr("PIL.Image.Image.save", lambda self, path: None)
    FakeDiffusion.last_request = None

    response = client.post("/generate", json={})

    assert response.status_code == 200
    assert response.json()["out"] == DEFAULT_GENERATE_REQUEST_EXAMPLE["output_path"]
    assert FakeDiffusion.last_request is not None
    assert FakeDiffusion.last_request.model == DEFAULT_GENERATE_REQUEST_EXAMPLE["model"]
    assert FakeDiffusion.last_request.prompt == DEFAULT_GENERATE_REQUEST_EXAMPLE["prompt"]
    assert (
        FakeDiffusion.last_request.negative_prompt
        == DEFAULT_GENERATE_REQUEST_EXAMPLE["negative_prompt"]
    )
    assert FakeDiffusion.last_request.width == DEFAULT_GENERATE_REQUEST_EXAMPLE["width"]
    assert FakeDiffusion.last_request.height == DEFAULT_GENERATE_REQUEST_EXAMPLE["height"]
    assert FakeDiffusion.last_request.steps == DEFAULT_GENERATE_REQUEST_EXAMPLE["steps"]
    assert (
        FakeDiffusion.last_request.guidance_scale
        == DEFAULT_GENERATE_REQUEST_EXAMPLE["guidance_scale"]
    )
    assert FakeDiffusion.last_request.seed == DEFAULT_GENERATE_REQUEST_EXAMPLE["seed"]
    assert FakeDiffusion.last_request.device == DEFAULT_GENERATE_REQUEST_EXAMPLE["device"]
    assert FakeDiffusion.last_request.dtype == DEFAULT_GENERATE_REQUEST_EXAMPLE["dtype"]


def test_generate_uses_default_payload_when_body_is_omitted(monkeypatch) -> None:
    monkeypatch.setattr("service.fastapi_app.main.Diffusion", FakeDiffusion)
    monkeypatch.setattr("PIL.Image.Image.save", lambda self, path: None)
    FakeDiffusion.last_request = None

    response = client.post("/generate")

    assert response.status_code == 200
    assert response.json()["out"] == DEFAULT_GENERATE_REQUEST_EXAMPLE["output_path"]
    assert FakeDiffusion.last_request is not None
    assert FakeDiffusion.last_request.model == DEFAULT_GENERATE_REQUEST_EXAMPLE["model"]
