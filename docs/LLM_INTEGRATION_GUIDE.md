# LLM Integration Guide

Guia curto para um modelo de linguagem que precise instalar `unified_diffusion` e usá-lo como componente dentro de uma aplicação terceira.

## Objetivo

Use este package quando a aplicação precisar:

- listar modelos disponíveis
- gerar imagem a partir de texto
- usar modelos embutidos ou modelos definidos em registry externo
- manter cache local determinístico

Não trate este package como:

- sistema de treino
- gerenciador de LoRA
- servidor remoto
- workflow engine completo

## Instalação em aplicação terceira

### Opção 1. Instalar a partir do repositório local

Se a aplicação terceira estiver na mesma máquina e puder apontar para este repositório:

```bash
pip install /caminho/para/unified_diffusion
```

Se precisar de suporte a FLUX:

```bash
pip install "/caminho/para/unified_diffusion[flux]"
```

### Opção 2. Desenvolvimento local com `uv`

Se a aplicação terceira usa `uv`:

```bash
uv add /caminho/para/unified_diffusion
```

Com extra opcional:

```bash
uv add "/caminho/para/unified_diffusion[flux]"
```

## Imports públicos permitidos

Ao integrar, use apenas os imports públicos:

```python
from unified_diffusion import Diffusion, GenerateRequest, GenerateResult
from unified_diffusion import ModelNotFoundError, ProviderError, CacheError
```

Não dependa diretamente de módulos internos como:

- `unified_diffusion.providers.*`
- `unified_diffusion.cache.*`
- `unified_diffusion.registry.*`

Esses módulos são internos e podem evoluir sem a mesma estabilidade da API pública.

## Uso mínimo dentro da aplicação

Exemplo simples:

```python
from unified_diffusion import Diffusion, GenerateRequest


engine = Diffusion(cache_dir="~/.cache/unified-diffusion")

result = engine.run(
    GenerateRequest(
        model="sdxl.base",
        prompt="a cinematic product photo, studio lighting, clean background",
        width=1024,
        height=1024,
        steps=30,
        guidance_scale=6.5,
        seed=1234,
        device="mps",
    )
)

result.images[0].save("out.png")
```

## Como usar como componente

Padrão recomendado:

1. crie uma instância de `Diffusion` no startup da aplicação
2. reuse essa instância para requests de geração
3. mantenha `cache_dir` fora do repositório da aplicação
4. trate exceções de modelo e provider na borda da aplicação

Exemplo de serviço interno:

```python
from pathlib import Path

from unified_diffusion import Diffusion, GenerateRequest, ModelNotFoundError, ProviderError


class ImageGenerationService:
    def __init__(self) -> None:
        self.engine = Diffusion(cache_dir="~/.cache/unified-diffusion")

    def generate(self, model: str, prompt: str, output_path: str) -> str:
        try:
            result = self.engine.run(
                GenerateRequest(
                    model=model,
                    prompt=prompt,
                    width=1024,
                    height=1024,
                    steps=30,
                    guidance_scale=6.5,
                )
            )
        except ModelNotFoundError as exc:
            raise ValueError(f"Unknown model: {model}") from exc
        except ProviderError as exc:
            raise RuntimeError(f"Generation failed: {exc}") from exc

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(out)
        return str(out)
```

## Como listar modelos

```python
from unified_diffusion import Diffusion

engine = Diffusion()
models = engine.list_models()
print(models)
```

## Como usar registry externo

Se a aplicação terceira precisar modelos customizados, defina:

```bash
export UNIFIED_DIFFUSION_REGISTRY_PATH=/caminho/para/custom-models.json
```

Formato mínimo:

```json
{
  "local.my.model": {
    "provider": "diffusers",
    "source": "/abs/path/to/model.safetensors",
    "pipeline_type": "stable-diffusion-xl"
  }
}
```

Regra prática:

- SDXL 1.0 e derivados: `stable-diffusion-xl`
- SD 1.x / 2.x: `stable-diffusion`
- SD3: `stable-diffusion-3`
- FLUX: `flux`

## Boas práticas para a aplicação terceira

- mantenha o cache em `~/.cache/unified-diffusion` ou outro diretório externo ao repo
- não copie pesos para dentro do repositório da aplicação
- para modelos locais, prefira `~/models/civitai/<slug>/model.safetensors`
- use SHA-256 do Civitai para validar arquivos antes do registro
- para checkpoints SDXL 1.0 locais, use `pipeline_type="stable-diffusion-xl"`
- quando um modelo default quebrar upstream, atualize o registry embutido deste projeto em `unified_diffusion/registry/models.py`

## Quando usar a CLI em vez da API

Use a CLI quando a aplicação terceira:

- estiver sendo prototipada manualmente
- precisar registrar modelos locais
- quiser validar um arquivo antes de integrar

Comandos úteis:

```bash
uv run udiff models
uv run udiff verify-file --path ~/Downloads/model.safetensors --sha256 HASH
uv run udiff register-local --path ~/Downloads/model.safetensors --sha256 HASH
uv run udiff guided-run --emit command
```

## Erros comuns

### Modelo não encontrado

Causa comum:

- `canonical_id` não existe
- `custom-models.json` não foi carregado

Ação:

- chame `Diffusion().list_models()`
- confirme `UNIFIED_DIFFUSION_REGISTRY_PATH`

### Erro de provider ao carregar `.safetensors`

Causa comum:

- arquivo corrompido
- hash divergente
- checkpoint incompatível com o loader single-file

Ação:

- valide com SHA-256
- teste com `udiff verify-file`
- rebaxe o arquivo se necessário

### Falha por dependência/pipeline

Causa comum:

- versão instalada do `diffusers` não expõe a classe necessária
- extra opcional não instalado

Ação:

- revisar dependências
- usar extra `flux` quando aplicável

## Instrução curta para outro modelo de linguagem

Se você for outro agente/modelo e precisar integrar este package:

1. use apenas a API pública exportada em `unified_diffusion`
2. instale o package no ambiente da aplicação terceira
3. crie um serviço fino que encapsule `Diffusion`
4. mantenha cache e pesos fora do repositório da aplicação
5. use registry externo para modelos customizados
6. valide arquivos locais com SHA-256 antes de registrá-los
