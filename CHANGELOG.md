# Changelog

All notable changes to this project should be documented in this file.

The format is intentionally simple and lightweight.

## [0.1.0] - 2026-03-21

### Added

- Minimal public Python API with `Diffusion`, `GenerateRequest`, and `GenerateResult`
- Built-in registry for core diffusion models
- External JSON registry support via `UNIFIED_DIFFUSION_REGISTRY_PATH`
- Deterministic cache layout and registry JSONL logging
- CLI with `run`, `models`, `guided-run`, `register-local`, `verify-file`, `practices`, and `cache ls`
- SHA-256 verification for local `.safetensors` registration
- FastAPI HTTP layer with Docker and local startup flows
- Quick reference and integration documentation

### Changed

- Documentation reorganized under `docs/`
- API docs improved with OpenAPI examples and error examples

### Notes

- This is the first documented release baseline for the repository.
