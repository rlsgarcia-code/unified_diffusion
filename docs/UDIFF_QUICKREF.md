# UDIFF Quick Reference

Guia curto para uso diário do `udiff`.

## Regras de layout

- pesos baixados dos modelos default ficam em `~/.cache/unified-diffusion`
- pesos customizados locais devem ficar fora do repo, idealmente em `~/models/civitai/<slug>/model.safetensors`
- o `udiff register-local` já usa esse layout por default
- evite apontar `--cache-dir` para dentro do repositório

## Help

```bash
uv run udiff --help
uv run udiff run --help
uv run udiff guided-run --help
uv run udiff register-local --help
uv run udiff models --help
uv run udiff verify-file --help
uv run udiff practices
```

## Listar modelos

```bash
uv run udiff models
```

Com registry externo:

```bash
export UNIFIED_DIFFUSION_REGISTRY_PATH=$PWD/custom-models.json
uv run udiff models
```

Se `custom-models.json` estiver no diretório atual, `udiff` tenta usar esse arquivo automaticamente mesmo sem `export`.

## Rodar imagem

O `udiff run` agora usa o mesmo default do endpoint HTTP `/generate`. Sem argumentos, ele assume:

- `model=sdxl.base`
- `prompt="a dramatic product photo, studio lighting, clean background"`
- `negative_prompt="blurry, low quality"`
- `width=1024`
- `height=1024`
- `steps=30`
- `guidance_scale=6.5`
- `seed=1234`
- `device=mps`
- `dtype=fp16`
- `out=/Users/robinsongarcia/Desktop/out.png`

Modo totalmente default:

```bash
uv run udiff run
```

Modo direto:

```bash
uv run udiff run \
  --model sdxl.base \
  --prompt "a cinematic portrait, soft light, 85mm lens" \
  --out out.png
```

Modo guiado:

```bash
uv run udiff guided-run
```

Só gerar o comando:

```bash
uv run udiff guided-run --emit command
```

O comando emitido agora inclui explicitamente os mesmos defaults do HTTP e do `udiff run`.

## Registrar modelo local

Registrar um arquivo `.safetensors`, mover para `~/models/civitai/<slug>/model.safetensors` e atualizar o JSON:

```bash
uv run udiff register-local --path ~/Downloads/model.safetensors
```

Com verificação de integridade via hash SHA-256 do Civitai:

```bash
uv run udiff register-local \
  --path ~/Downloads/model.safetensors \
  --sha256 HASH_DO_CIVITAI
```

Validar sem mover nem registrar:

```bash
uv run udiff verify-file \
  --path ~/Downloads/model.safetensors \
  --sha256 HASH_DO_CIVITAI
```

Fluxo esperado:

1. informar ou aceitar o `slug`
2. informar ou aceitar o `canonical_id`
3. aceitar defaults como `diffusers` e `stable-diffusion-xl` se fizer sentido
4. revisar o `sample_command` exibido ao final

## Cache

```bash
uv run udiff cache ls
```

## Defaults úteis

- `provider`: `diffusers`
- `pipeline_type`: `stable-diffusion-xl`
- `output directory` no modo guiado: `outputs`
- `output file name` no modo guiado: `<model>.png`

## Best practices no CLI

```bash
uv run udiff practices
```

## Fluxo curto recomendado

```bash
export UNIFIED_DIFFUSION_REGISTRY_PATH=$PWD/custom-models.json
uv run udiff models
uv run udiff guided-run
```

## Se um modelo default quebrar

Edite o registry embutido em:

- [`unified_diffusion/registry/models.py`](/Users/robinsongarcia/projects/unified_diffusion/unified_diffusion/registry/models.py)

É ali que ficam os `source` e `default_revision` dos modelos default.
