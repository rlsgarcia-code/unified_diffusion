import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

from unified_diffusion import GenerateResult
from unified_diffusion.cli import (
    best_usage_practices,
    main,
    register_local_model,
    verify_local_file,
)
from unified_diffusion.registry.models import CUSTOM_REGISTRY_ENV


class FakeDiffusion:
    def __init__(self, cache_dir: str) -> None:
        self.cache = type(
            "CacheView",
            (),
            {
                "root": Path(cache_dir),
                "log_path": Path(cache_dir) / "logs" / "registry.jsonl",
                "list_cached_models": lambda self: ["sdxl.base/main"],
            },
        )()

    def list_models(self) -> list[str]:
        return ["sdxl.base", "local.civitai.omnigenx"]

    def run(self, request):
        image = Image.new("RGB", (request.width, request.height), color="white")
        return GenerateResult(
            images=[image],
            seed_used=request.seed,
            model_resolved=f"{request.model}@main",
            provider_used="diffusers",
            cache_path="/tmp/model",
            metadata={},
        )


def test_cli_run_writes_image(monkeypatch, tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "out.png"
    monkeypatch.setattr("unified_diffusion.cli.Diffusion", FakeDiffusion)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "udiff",
            "run",
            "--model",
            "sdxl.base",
            "--prompt",
            "hello",
            "--out",
            str(output_path),
            "--cache-dir",
            str(tmp_path),
        ],
    )

    exit_code = main()

    assert exit_code == 0
    assert output_path.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "sdxl.base@main"


def test_cli_cache_ls_prints_json(monkeypatch, tmp_path: Path, capsys) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "registry.jsonl").write_text(
        json.dumps({"canonical_id": "sdxl.base", "provider": "diffusers"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("unified_diffusion.cli.Diffusion", FakeDiffusion)
    monkeypatch.setattr(sys, "argv", ["udiff", "cache", "--cache-dir", str(tmp_path), "ls"])

    exit_code = main()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"] == ["sdxl.base/main"]
    assert payload["last_records"][0]["canonical_id"] == "sdxl.base"


def test_cli_models_prints_json(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("unified_diffusion.cli.Diffusion", FakeDiffusion)
    monkeypatch.setattr(sys, "argv", ["udiff", "models", "--cache-dir", str(tmp_path)])

    exit_code = main()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"] == ["sdxl.base", "local.civitai.omnigenx"]


def test_cli_models_auto_loads_custom_models_json_from_cwd(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    custom_registry = tmp_path / "custom-models.json"
    custom_registry.write_text(
        json.dumps(
            {
                "local.civitai.demo": {
                    "provider": "diffusers",
                    "source": "/tmp/demo/model.safetensors",
                    "pipeline_type": "stable-diffusion-xl",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(CUSTOM_REGISTRY_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", ["udiff", "models", "--cache-dir", str(tmp_path / "cache")])

    exit_code = main()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "local.civitai.demo" in payload["models"]


def test_register_local_model_moves_file_and_updates_registry(monkeypatch, tmp_path: Path) -> None:
    source_path = tmp_path / "Fancy Model v1.safetensors"
    source_path.write_text("stub", encoding="utf-8")
    registry_path = tmp_path / "custom-models.json"
    models_dir = tmp_path / "models"

    answers = iter(
        [
            "",
            "",
            "",
            "",
            "",
            "",
            "Test license",
            "Test notes",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    payload = register_local_model(
        source_path=source_path,
        registry_path=registry_path,
        models_dir=models_dir,
    )

    moved_to = models_dir / "fancy-model-v1" / "model.safetensors"
    assert moved_to.exists()
    assert not source_path.exists()
    assert payload["canonical_id"] == "local.civitai.fancy-model-v1"
    assert payload["moved_to"] == str(moved_to)
    assert "uv run udiff run --model local.civitai.fancy-model-v1" in payload["sample_command"]

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["local.civitai.fancy-model-v1"]["provider"] == "diffusers"
    assert registry["local.civitai.fancy-model-v1"]["pipeline_type"] == "stable-diffusion-xl"
    assert registry["local.civitai.fancy-model-v1"]["source"] == str(moved_to)
    assert registry["local.civitai.fancy-model-v1"]["license_hint"] == "Test license"
    assert payload["sha256_verified"] == "false"


def test_cli_register_local_prints_payload(monkeypatch, tmp_path: Path, capsys) -> None:
    source_path = tmp_path / "demo.safetensors"
    source_path.write_text("stub", encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    models_dir = tmp_path / "models"

    answers = iter(["demo-model", "local.demo-model", "", "", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "udiff",
            "register-local",
            "--path",
            str(source_path),
            "--registry-path",
            str(registry_path),
            "--models-dir",
            str(models_dir),
        ],
    )

    exit_code = main()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["canonical_id"] == "local.demo-model"
    assert payload["moved_to"] == str(models_dir / "demo-model" / "model.safetensors")


def test_register_local_model_verifies_sha256(monkeypatch, tmp_path: Path) -> None:
    source_path = tmp_path / "sha.safetensors"
    source_path.write_bytes(b"abc123")
    registry_path = tmp_path / "custom-models.json"
    models_dir = tmp_path / "models"
    expected_sha256 = hashlib.sha256(b"abc123").hexdigest()

    answers = iter(["", "", "", "", "", "Test license", "Test notes"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    payload = register_local_model(
        source_path=source_path,
        registry_path=registry_path,
        models_dir=models_dir,
        expected_sha256=expected_sha256,
    )

    assert payload["sha256_verified"] == "true"
    assert payload["sha256_actual"] == expected_sha256


def test_register_local_model_rejects_sha256_mismatch(monkeypatch, tmp_path: Path) -> None:
    source_path = tmp_path / "badsha.safetensors"
    source_path.write_bytes(b"abc123")
    registry_path = tmp_path / "custom-models.json"
    models_dir = tmp_path / "models"

    answers = iter(["", "", "", "", "", "Test license", "Test notes"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    try:
        register_local_model(
            source_path=source_path,
            registry_path=registry_path,
            models_dir=models_dir,
            expected_sha256="0" * 64,
        )
        raise AssertionError("Expected SHA-256 mismatch to raise")
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    assert source_path.exists()
    assert not registry_path.exists()


def test_verify_local_file_returns_sha256(tmp_path: Path) -> None:
    source_path = tmp_path / "verify.safetensors"
    source_path.write_bytes(b"abc123")

    payload = verify_local_file(source_path=source_path)

    assert payload["file_name"] == "verify.safetensors"
    assert payload["sha256"] == hashlib.sha256(b"abc123").hexdigest()
    assert payload["sha256_verified"] == "false"


def test_cli_verify_file_prints_payload(tmp_path: Path, capsys, monkeypatch) -> None:
    source_path = tmp_path / "verifycli.safetensors"
    source_path.write_bytes(b"abc123")
    expected_sha256 = hashlib.sha256(b"abc123").hexdigest()
    monkeypatch.setattr(
        sys,
        "argv",
        ["udiff", "verify-file", "--path", str(source_path), "--sha256", expected_sha256],
    )

    exit_code = main()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sha256_verified"] == "true"
    assert payload["file_name"] == "verifycli.safetensors"


def test_cli_practices_prints_best_practices(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["udiff", "practices"])

    exit_code = main()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["practices"] == best_usage_practices()


def test_cli_guided_run_command_mode(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("unified_diffusion.cli.Diffusion", FakeDiffusion)
    answers = iter(
        [
            "2",
            "portrait photo of a person, studio lighting",
            "",
            "",
            "command",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(sys, "argv", ["udiff", "guided-run", "--cache-dir", str(tmp_path)])

    exit_code = main()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "command"
    assert payload["model"] == "local.civitai.omnigenx"
    assert "--model local.civitai.omnigenx" in payload["command"]
    assert payload["out"].endswith("outputs/local_civitai_omnigenx.png")


def test_cli_guided_run_executes(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("unified_diffusion.cli.Diffusion", FakeDiffusion)
    answers = iter(
        [
            "sdxl.base",
            "hello world",
            str(tmp_path / "renders"),
            "hello.png",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(
        sys,
        "argv",
        ["udiff", "guided-run", "--cache-dir", str(tmp_path), "--emit", "run"],
    )

    exit_code = main()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "run"
    assert payload["resolved_model"] == "sdxl.base@main"
    assert Path(payload["out"]).exists()
