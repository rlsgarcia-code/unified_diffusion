# API Quick Reference

Guia curto para subir e usar a API HTTP do `unified_diffusion`.

## Modos de execução

Use Docker quando você quiser isolamento e um fluxo mais reproduzível.

Use execução local quando quiser aproveitar aceleração da máquina host, especialmente `mps` no macOS Apple Silicon.

## Comandos de conveniência

Com `npm run`:

```bash
npm run up:docker
npm run down:docker
npm run logs:docker
npm run up:api
npm run test:api
```

## Subir com Docker

```bash
npm run up:docker
```

Comando equivalente:

```bash
docker compose -f docker-compose.api.yml up --build
```

O compose expõe:

- API em `http://127.0.0.1:8000`
- cache em `${HOME}/.cache/unified-diffusion`
- modelos locais em `${HOME}/models`
- outputs em `./outputs`

## Subir fora do Docker

Este é o caminho recomendado quando você quer aproveitar `mps`.

Primeiro, garanta o ambiente:

```bash
uv sync
```

Se for usar FLUX:

```bash
uv sync --extra flux
```

Depois suba:

```bash
npm run up:api
```

Comando equivalente:

```bash
uv run uvicorn service.fastapi_app.main:app --host 0.0.0.0 --port 8000
```

Se quiser forçar registry customizado:

```bash
UNIFIED_DIFFUSION_REGISTRY_PATH=$PWD/custom-models.json npm run up:api
```

Ou sem `npm`:

```bash
UNIFIED_DIFFUSION_REGISTRY_PATH=$PWD/custom-models.json \
uv run uvicorn service.fastapi_app.main:app --host 0.0.0.0 --port 8000
```

## Health check

```bash
curl -sS http://127.0.0.1:8000/health
```

## Docs interativas

Swagger UI:

```bash
open http://127.0.0.1:8000/docs
```

OpenAPI JSON:

```bash
curl -sS http://127.0.0.1:8000/openapi.json
```

Os schemas da API agora incluem exemplos prontos para:

- `/generate`
- `/verify-file`
- `/register-local`

## Listar modelos

```bash
curl -sS http://127.0.0.1:8000/models
```

## Ver práticas recomendadas

```bash
curl -sS http://127.0.0.1:8000/practices
```

## Verificar um arquivo local

```bash
curl -sS http://127.0.0.1:8000/verify-file \
  -H 'Content-Type: application/json' \
  -d '{
    "path": "/Users/seu-user/Downloads/model.safetensors",
    "sha256": "HASH_DO_CIVITAI"
  }'
```

## Registrar modelo local

```bash
curl -sS http://127.0.0.1:8000/register-local \
  -H 'Content-Type: application/json' \
  -d '{
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
    "sha256": "HASH_DO_CIVITAI"
  }'
```

## Gerar imagem

Exemplo local, ideal para usar `mps`:

```bash
curl -sS http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sdxl.base",
    "prompt": "a dramatic product photo, studio lighting, clean background",
    "device": "mps",
    "dtype": "fp16",
    "output_path": "/Users/seu-user/Desktop/out.png"
  }'
```

Exemplo com modelo custom:

```bash
curl -sS http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local.civitai.omnigenx",
    "prompt": "portrait photo, studio lighting, high detail",
    "device": "mps",
    "dtype": "fp16",
    "output_path": "/Users/seu-user/Desktop/omnigenx.png"
  }'
```

## Regras práticas

- fora do Docker, prefira `npm run up:api` para usar `mps`
- dentro do Docker, trate o container como fluxo CPU-first
- use `verify-file` com SHA-256 antes de registrar arquivos do Civitai
- mantenha pesos fora do repositório
- mantenha `UNIFIED_DIFFUSION_REGISTRY_PATH` apontando para um JSON explícito quando usar modelos customizados
