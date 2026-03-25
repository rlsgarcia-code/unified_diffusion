# unified_diffusion

`unified_diffusion` fornece uma abstração Python mínima para rodar diferentes modelos de difusão com uma API estável, cache determinístico e download lazy no primeiro uso.

## Instalação

```bash
uv sync
```

Sincronizar com extras opcionais:

```bash
uv sync --extra flux
```

O repositório pode ser reproduzido em outra máquina com [`pyproject.toml`](/Users/robinsongarcia/projects/unified_diffusion/pyproject.toml) e `uv.lock`.

`accelerate` faz parte do ambiente base para melhorar carregamento e uso de memória nos pipelines `diffusers`.

Artefatos de build locais como `*.egg-info/` não fazem parte do repositório e ficam ignorados.

## Exemplo mínimo

```python
from unified_diffusion import Diffusion, GenerateRequest

engine = Diffusion(cache_dir="~/.cache/unified-diffusion")

result = engine.run(
    GenerateRequest(
        model="sdxl.base",
        prompt="a cinematic photo of a vintage racing car, 35mm, shallow depth of field",
        negative_prompt="blurry, low quality",
        width=1024,
        height=1024,
        steps=30,
        guidance_scale=6.5,
        seed=1234,
        device="mps",
    )
)

result.images[0].save("out.png")
print(result.model_resolved, result.provider_used)
```

## Modelos suportados

```python
from unified_diffusion import Diffusion

engine = Diffusion()
print(engine.list_models())
```

Modelos MVP:

- `sdxl.base`
- `sdxl.refiner`
- `sd21.base`
- `playground.v25`
- `pixart.alpha`
- `pixart.sigma`
- `sd3.medium`
- `flux.1-dev`

Alguns modelos dependem da versão instalada de `diffusers` ou de extras opcionais. Quando o ambiente não suportar um modelo, a biblioteca retorna um erro amigável explicando como habilitá-lo.

Guia detalhado de uso dos modelos:

- [`MODELS.md`](/Users/robinsongarcia/projects/unified_diffusion/docs/MODELS.md)
- [`ADDING_MODELS.md`](/Users/robinsongarcia/projects/unified_diffusion/docs/ADDING_MODELS.md)
- [`CIVITAI.md`](/Users/robinsongarcia/projects/unified_diffusion/docs/CIVITAI.md)
- [`UDIFF_QUICKREF.md`](/Users/robinsongarcia/projects/unified_diffusion/docs/UDIFF_QUICKREF.md)
- [`LLM_INTEGRATION_GUIDE.md`](/Users/robinsongarcia/projects/unified_diffusion/docs/LLM_INTEGRATION_GUIDE.md)
- [`API_QUICKREF.md`](/Users/robinsongarcia/projects/unified_diffusion/docs/API_QUICKREF.md)
- [`docs/README.md`](/Users/robinsongarcia/projects/unified_diffusion/docs/README.md)
- [`CHANGELOG.md`](/Users/robinsongarcia/projects/unified_diffusion/CHANGELOG.md)
- [`RELEASE_PROCESS.md`](/Users/robinsongarcia/projects/unified_diffusion/docs/RELEASE_PROCESS.md)

## API HTTP e Swagger

O repositório também expõe uma API HTTP via FastAPI.

Suba a API:

```bash
npm run up:api
```

Ou:

```bash
uv run uvicorn service.fastapi_app.main:app --host 0.0.0.0 --port 8000
```

Depois abra no navegador:

- raiz da API: `http://127.0.0.1:8000/`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

A raiz agora redireciona para Swagger UI, e o `/docs` permite testar todos os endpoints interativamente, incluindo:

- `/models`
- `/practices`
- `/verify-file`
- `/register-local`
- `/generate`

O fluxo recomendado para modelo local via Swagger é:

1. chamar `/verify-file` com o caminho do `.safetensors` e, se tiver, o SHA-256
2. chamar `/register-local` para mover o arquivo para `~/models/civitai/<slug>/model.safetensors` e atualizar o registry JSON
3. confirmar o novo `canonical_id` em `/models`
4. usar o `canonical_id` em `/generate`

Guia curto da API:

- [`API_QUICKREF.md`](/Users/robinsongarcia/projects/unified_diffusion/docs/API_QUICKREF.md)


## Registry customizado

Você pode registrar modelos adicionais sem alterar o código do package usando um arquivo JSON externo apontado por `UNIFIED_DIFFUSION_REGISTRY_PATH`.

### Como pesquisar modelos

Fluxo recomendado:

1. Pesquise no Hugging Face por modelos `diffusers` ou por um `repo_id` que tenha suporte documentado ao pipeline desejado.
2. Abra o model card e confirme:
   - qual é o `repo_id` exato
   - se o modelo exige aceite de licença/gating
   - se o formato é compatível com `diffusers`
   - qual pipeline faz sentido: `stable-diffusion`, `stable-diffusion-xl`, `stable-diffusion-3`, `pixart-alpha`, `pixart-sigma` ou `flux`
3. Verifique se a versão local do `diffusers` expõe a classe de pipeline correspondente.
4. Só então adicione a entrada no JSON.

Heurística prática:

- modelos SD 1.x / 2.x costumam mapear para `stable-diffusion`
- SDXL e derivados costumam mapear para `stable-diffusion-xl`
- SD3 costuma mapear para `stable-diffusion-3`
- PixArt Alpha/Sigma mapeiam para `pixart-alpha` e `pixart-sigma`
- FLUX mapeia para `flux`

### Campos do JSON

- `provider`: backend interno que o package vai usar. Hoje os valores úteis são `diffusers` e `flux`.
- `source`: pode ser um `repo_id` do Hugging Face, um diretório local ou um arquivo local `.safetensors`.
- `pipeline_type`: tipo lógico do pipeline. Ele orienta a seleção da classe de pipeline no provider.
- `default_revision`: opcional. Revision/branch/tag padrão. Se omitido, o package usa `main`.
- `license_hint`: opcional. Texto livre, só para documentação e logging.
- `notes`: opcional. Texto livre com observações práticas.

### Pesos locais

O package agora aceita `source` local de duas formas:

1. diretório local compatível com `diffusers`
2. arquivo único `.safetensors`

Exemplo com diretório local:

```json
{
  "local.my.sdxl.dir": {
    "provider": "diffusers",
    "source": "~/models/civitai/my-sdxl-dir",
    "pipeline_type": "stable-diffusion-xl"
  }
}
```

Exemplo com arquivo `.safetensors`:

```json
{
  "local.my.sdxl.file": {
    "provider": "diffusers",
    "source": "~/models/civitai/my-sdxl/model.safetensors",
    "pipeline_type": "stable-diffusion-xl"
  }
}
```

`.safetensors` pode ser checkpoint válido, sim. A ressalva é técnica:

- se você tiver um diretório completo no formato esperado pelo `diffusers`, esse é o caminho mais seguro
- se você tiver só um arquivo `.safetensors`, ele pode funcionar quando o pipeline suporta `from_single_file()`
- se não suportar, o arquivo sozinho não basta

### Exemplo comentado

```json
{
  "research.custom.one": {
    "provider": "diffusers",
    "source": "org-or-user/model-one",
    "pipeline_type": "stable-diffusion-xl",
    "default_revision": "main",
    "license_hint": "Check the upstream license before use.",
    "notes": "Private or experimental model."
  }
}
```

Exemplo:

```json
{
  "research.custom.one": {
    "provider": "diffusers",
    "source": "org-or-user/model-one",
    "pipeline_type": "stable-diffusion-xl",
    "default_revision": "main",
    "license_hint": "Check the upstream license before use.",
    "notes": "Private or experimental model."
  },
  "research.custom.two": {
    "provider": "diffusers",
    "source": "org-or-user/model-two",
    "pipeline_type": "stable-diffusion"
  }
}
```

Uso:

```bash
export UNIFIED_DIFFUSION_REGISTRY_PATH=$PWD/custom-models.json
uv run python -c "from unified_diffusion import Diffusion; print(Diffusion().list_models())"
```

Na CLI, se você estiver na raiz do projeto e existir um `custom-models.json`, o `udiff` também detecta esse arquivo automaticamente.

Template pronto:

- [`custom-models.example.json`](/Users/robinsongarcia/projects/unified_diffusion/custom-models.example.json)

Para FLUX, instale preferencialmente:

```bash
uv sync --extra flux
```

O provider tenta usar `FluxPipeline` via `diffusers`. Se a versão instalada não expuser esse pipeline, o erro orienta a atualizar as dependências.

## Cache

O cache padrão fica em `~/.cache/unified-diffusion` e segue este layout:

```text
models/<canonical_id>/<revision>/
tmp/
logs/registry.jsonl
```

O download é lazy: pesos só são baixados no primeiro uso de cada modelo. O pacote força o uso de um cache único para Hugging Face dentro do `cache_dir`, evitando caches espalhados fora desse diretório.

Por padrão, isso mantém os pesos baixados fora do repositório clonado. Para o fluxo normal, não use `--cache-dir` apontando para dentro do repo.

Para modelos customizados locais, o caminho recomendado também fica fora do repo:

```text
~/models/civitai/<slug>/model.safetensors
```

O subcomando `udiff register-local` já usa esse layout por default.

Quando `device="mps"` for pedido explicitamente e MPS não estiver disponível, a biblioteca retorna erro. Quando o dispositivo vier por default e nenhum acelerador estiver disponível, o provider pode cair para CPU e registrar o warning em `GenerateResult.metadata["warnings"]`.

## Onde editar fontes dos modelos default

Quando alguém clona o repositório e roda um modelo embutido, o `udiff` baixa o peso automaticamente no primeiro uso a partir do registry interno.

Se algum `repo_id` default do Hugging Face mudar, quebrar ou precisar de revisão nova, o lugar correto para editar é:

- [`unified_diffusion/registry/models.py`](/Users/robinsongarcia/projects/unified_diffusion/unified_diffusion/registry/models.py)

Cada entrada do `MODEL_REGISTRY` define:

- `canonical_id`
- `source`
- `default_revision`
- `pipeline_type`

Esse é o ponto único apropriado para manutenção dos links dos modelos embutidos.

## CLI

Ajuda geral:

```bash
uv run udiff --help
```

Listar modelos resolvidos pelo registry atual:

```bash
uv run udiff models
```

## API e smoke tests

Subir a API localmente:

```bash
npm run up:api
```

Smoke local da API:

```bash
npm run smoke:api:local
```

Smoke via Docker:

```bash
npm run smoke:api:docker
```

Os workflows de CI e release executam os smokes local e Docker antes de considerar a build válida.

Rodar em modo direto:

```bash
uv run udiff run --model sdxl.base --prompt "a dramatic product photo" --out out.png
```

Rodar em modo guiado:

```bash
uv run udiff guided-run
```

Rodar o assistente guiado e pedir apenas o comando final:

```bash
uv run udiff guided-run --emit command
```

Registrar um `.safetensors` local, mover para o layout padronizado e atualizar o registry:

```bash
uv run udiff register-local --path ~/Downloads/model.safetensors
```

Inspecionar o cache:

```bash
uv run udiff cache ls
```

Referência curta de uso:

- [`UDIFF_QUICKREF.md`](/Users/robinsongarcia/projects/unified_diffusion/docs/UDIFF_QUICKREF.md)

## API HTTP

O repositório também inclui uma FastAPI fina sobre o package para expor:

- `GET /health`
- `GET /models`
- `GET /practices`
- `POST /verify-file`
- `POST /register-local`
- `POST /generate`

Guia curto da API:

- [`API_QUICKREF.md`](/Users/robinsongarcia/projects/unified_diffusion/docs/API_QUICKREF.md)

### Scripts de conveniência

```bash
npm run up:docker
npm run down:docker
npm run logs:docker
npm run up:api
npm run test:api
```

### Quando usar Docker

Use Docker quando você quiser:

- isolamento de ambiente
- fluxo reprodutível
- execução CPU-first

Subir com Docker:

```bash
npm run up:docker
```

### Quando usar fora do Docker

Use fora do Docker quando quiser aproveitar aceleração da máquina host, especialmente:

- `mps` no macOS Apple Silicon

Subir localmente:

```bash
uv sync
npm run up:api
```

Se quiser usar modelos customizados:

```bash
UNIFIED_DIFFUSION_REGISTRY_PATH=$PWD/custom-models.json npm run up:api
```

### Observações operacionais da API

- no modo local, prefira cache fora do repo
- no modo Docker, o compose monta cache, modelos e outputs por volume
- para arquivos do Civitai, prefira `verify-file` com SHA-256 antes de `register-local`

### Exemplos rápidos com curl

Health:

```bash
curl -sS http://127.0.0.1:8000/health
```

Listar modelos:

```bash
curl -sS http://127.0.0.1:8000/models
```

Verificar arquivo local:

```bash
curl -sS http://127.0.0.1:8000/verify-file \
  -H 'Content-Type: application/json' \
  -d '{
    "path": "/Users/seu-user/Downloads/model.safetensors",
    "sha256": "HASH_DO_CIVITAI"
  }'
```

Gerar imagem:

```bash
curl -sS http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sdxl.base",
    "prompt": "a dramatic product photo, studio lighting",
    "device": "mps",
    "dtype": "fp16",
    "output_path": "/Users/seu-user/Desktop/out.png"
  }'
```

## Desenvolvimento

```bash
uv run pytest
uv run ruff check .
```

## Aviso de licenças

Alguns modelos possuem licenças não comerciais ou restrições de uso. Verifique a licença do modelo antes de utilizar em produção ou comercialmente. Este projeto não faz aconselhamento jurídico.
