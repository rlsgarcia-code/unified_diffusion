from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

from unified_diffusion import Diffusion, GenerateRequest
from unified_diffusion.operations import (
    best_usage_practices,
    configure_registry_path,
    register_local_model_entry,
    verify_local_file,
)
from unified_diffusion.settings import get_settings


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="udiff",
        description="Unified diffusion CLI for running and registering local models.",
        epilog=(
            "Best-practice reminders: run `udiff practices` and "
            "verify custom files with `udiff verify-file`."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Generate an image")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--prompt", required=True)
    run_parser.add_argument("--negative-prompt")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--width", type=int, default=1024)
    run_parser.add_argument("--height", type=int, default=1024)
    run_parser.add_argument("--steps", type=int, default=30)
    run_parser.add_argument("--guidance-scale", type=float, default=7.0)
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--device")
    run_parser.add_argument("--dtype")
    run_parser.add_argument("--cache-dir", default=str(settings.cache_dir))
    run_parser.add_argument("--registry-path")

    guided_run_parser = subparsers.add_parser(
        "guided-run",
        help="Interactively select a model and either run it or print the command",
    )
    guided_run_parser.add_argument("--cache-dir", default=str(settings.cache_dir))
    guided_run_parser.add_argument("--registry-path")
    guided_run_parser.add_argument(
        "--emit",
        choices=["run", "command"],
        help="Optional shortcut to skip the final mode prompt.",
    )

    models_parser = subparsers.add_parser("models", help="List available model ids")
    models_parser.add_argument("--cache-dir", default=str(settings.cache_dir))
    models_parser.add_argument("--registry-path")

    verify_parser = subparsers.add_parser(
        "verify-file",
        help="Verify a local .safetensors file before registering or running",
    )
    verify_parser.add_argument("--path", required=True)
    verify_parser.add_argument("--sha256", help="Expected SHA-256 from Civitai or another source")

    subparsers.add_parser("practices", help="Show best usage practices for udiff")

    register_parser = subparsers.add_parser(
        "register-local",
        help=(
            "Register a local .safetensors model interactively "
            "and move it into the standard layout"
        ),
    )
    register_parser.add_argument("--path", required=True)
    register_parser.add_argument("--registry-path", default=str(settings.default_registry_path))
    register_parser.add_argument("--models-dir", default=str(settings.default_models_dir))
    register_parser.add_argument(
        "--sha256",
        help="Expected SHA-256 for integrity verification before moving",
    )

    cache_parser = subparsers.add_parser("cache", help="Inspect local cache")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", required=True)
    cache_subparsers.add_parser("ls", help="List cache contents")
    cache_parser.add_argument("--cache-dir", default=str(settings.cache_dir))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_registry_path(getattr(args, "registry_path", None))

    if args.command == "run":
        engine = Diffusion(cache_dir=args.cache_dir)
        result = engine.run(
            GenerateRequest(
                model=args.model,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                device=args.device,
                dtype=args.dtype,
            )
        )
        output_path = Path(args.out).expanduser()
        result.images[0].save(output_path)
        print(
            json.dumps(
                {
                    "out": str(output_path),
                    "model": result.model_resolved,
                    "provider": result.provider_used,
                }
            )
        )
        return 0

    if args.command == "register-local":
        payload = register_local_model(
            source_path=Path(args.path).expanduser(),
            registry_path=Path(args.registry_path).expanduser(),
            models_dir=Path(args.models_dir).expanduser(),
            expected_sha256=args.sha256,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "verify-file":
        payload = verify_local_file(
            source_path=Path(args.path).expanduser(),
            expected_sha256=args.sha256,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "guided-run":
        engine = Diffusion(cache_dir=args.cache_dir)
        payload = guided_run(engine=engine, emit=args.emit)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "models":
        engine = Diffusion(cache_dir=args.cache_dir)
        print(json.dumps({"models": engine.list_models()}, indent=2, sort_keys=True))
        return 0

    if args.command == "practices":
        print(json.dumps({"practices": best_usage_practices()}, indent=2, sort_keys=True))
        return 0

    if args.command == "cache" and args.cache_command == "ls":
        engine = Diffusion(cache_dir=args.cache_dir)
        payload = {
            "models": engine.cache.list_cached_models(),
            "log_path": str(engine.cache.log_path),
            "last_records": _tail_jsonl(engine.cache.log_path, limit=5),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    parser.error("Unknown command")
    return 2


def _tail_jsonl(path: Path, limit: int) -> list[dict[str, object]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines[-limit:]:
        records.append(json.loads(line))
    return records


def register_local_model(
    source_path: Path,
    registry_path: Path,
    models_dir: Path,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    if not source_path.exists():
        raise FileNotFoundError(f"Model file not found: {source_path}")
    if source_path.suffix.lower() != ".safetensors":
        raise ValueError(f"Expected a .safetensors file, got: {source_path.name}")

    default_slug = _slugify(source_path.stem)
    slug = _prompt_with_default("Model folder slug", default_slug)
    canonical_id = _prompt_with_default("Canonical model id", f"local.civitai.{slug}")
    provider = _prompt_with_default("Provider", "diffusers")
    pipeline_type = _prompt_with_default("Pipeline type", "stable-diffusion-xl")
    default_revision = _prompt_with_default("Default revision", "main")
    sha256_value = expected_sha256 or _prompt_with_default("Civitai SHA256 (optional)", "")
    license_hint = _prompt_with_default("License hint", "__PREENCHER__")
    notes = _prompt_with_default("Notes", "Local Civitai checkpoint.")
    return register_local_model_entry(
        source_path=source_path,
        registry_path=registry_path,
        models_dir=models_dir,
        model_slug=slug,
        canonical_id=canonical_id,
        provider=provider,
        pipeline_type=pipeline_type,
        default_revision=default_revision,
        license_hint=license_hint,
        notes=notes,
        expected_sha256=sha256_value or None,
    )


def guided_run(engine: Diffusion, emit: str | None = None) -> dict[str, str]:
    model = _prompt_model_selection(engine.list_models())
    prompt = _prompt_non_empty("Prompt")
    output_dir = Path(_prompt_with_default("Output directory", "outputs")).expanduser()
    default_name = f"{model.replace('.', '_')}.png"
    output_name = _prompt_with_default("Output file name", default_name)
    output_path = output_dir / output_name
    emit_mode = emit or _prompt_choice("Mode", ["run", "command"], "run")

    command = _build_run_command(
        model=model,
        prompt=prompt,
        out=output_path,
        cache_dir=engine.cache.root,
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "out": str(output_path),
        "mode": emit_mode,
        "command": command,
    }
    if emit_mode == "command":
        return payload

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = engine.run(
        GenerateRequest(
            model=model,
            prompt=prompt,
        )
    )
    result.images[0].save(output_path)
    payload["provider"] = result.provider_used
    payload["resolved_model"] = result.model_resolved
    return payload


def _load_registry_object(registry_path: Path) -> dict[str, object]:
    if not registry_path.exists():
        return {}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Registry file must contain a JSON object: {registry_path}")
    return payload


def _prompt_with_default(label: str, default: str) -> str:
    answer = input(f"{label} [{default}]: ").strip()
    return answer or default


def _prompt_non_empty(label: str) -> str:
    while True:
        answer = input(f"{label}: ").strip()
        if answer:
            return answer
        print(f"{label} cannot be empty.", file=sys.stderr)


def _prompt_choice(label: str, options: list[str], default: str) -> str:
    options_label = "/".join(options)
    while True:
        answer = input(f"{label} [{default}] ({options_label}): ").strip().lower()
        value = answer or default
        if value in options:
            return value
        print(f"Invalid choice: {value}", file=sys.stderr)


def _prompt_model_selection(models: list[str]) -> str:
    if not models:
        raise ValueError("No models are available in the registry.")

    print("Available models:", file=sys.stderr)
    for index, model in enumerate(models, start=1):
        print(f"{index}. {model}", file=sys.stderr)

    while True:
        answer = input("Select model by number or canonical id: ").strip()
        if not answer:
            print("Model selection cannot be empty.", file=sys.stderr)
            continue
        if answer.isdigit():
            choice = int(answer)
            if 1 <= choice <= len(models):
                return models[choice - 1]
        elif answer in models:
            return answer
        print(f"Unknown model selection: {answer}", file=sys.stderr)


def _build_run_command(model: str, prompt: str, out: Path, cache_dir: Path) -> str:
    parts = [
        "uv",
        "run",
        "udiff",
        "run",
        "--model",
        model,
        "--prompt",
        prompt,
        "--out",
        str(out),
        "--cache-dir",
        str(cache_dir),
    ]
    return " ".join(shlex.quote(part) for part in parts)


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "model"


if __name__ == "__main__":
    raise SystemExit(main())
