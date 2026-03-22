# Adding New Models

Este documento explica como adicionar novos modelos ao `unified_diffusion` sem ambiguidade conceitual.

O objetivo é separar claramente:

- família do modelo
- variante
- modalidade
- modo de geração
- tipo de artefato
- formato de arquivo
- status editorial

## Resumo prático

Hoje o package suporta bem este caso:

- modelo base de imagem
- checkpoint/pipeline completo
- formato estrutural `diffusers`
- geração `text-to-image`

Em termos práticos, um novo modelo entra de duas formas:

1. registry externo via `UNIFIED_DIFFUSION_REGISTRY_PATH`
2. registry embutido em [`unified_diffusion/registry/models.py`](/Users/robinsongarcia/projects/unified_diffusion/unified_diffusion/registry/models.py)

Se o modelo continuar dentro do mesmo padrão técnico, normalmente basta:

- criar uma tag canônica
- apontar o `source`
- escolher o `provider`
- escolher o `pipeline_type`

`source` pode ser:

- `repo_id` remoto
- diretório local
- arquivo local `.safetensors`

## O que já existe na estrutura atual

Campos realmente usados hoje pelo runtime:

- `GenerateRequest.model`
- `GenerateRequest.provider`
- `GenerateRequest.revision`
- `GenerateRequest.device`
- `GenerateRequest.dtype`
- `GenerateRequest.extra`

Campos do registry realmente usados hoje:

- `canonical_id`
- `provider`
- `source`
- `pipeline_type`
- `default_revision`
- `license_hint`
- `notes`

Arquivos principais:

- API pública: [`unified_diffusion/api.py`](/Users/robinsongarcia/projects/unified_diffusion/unified_diffusion/api.py)
- Registry: [`unified_diffusion/registry/models.py`](/Users/robinsongarcia/projects/unified_diffusion/unified_diffusion/registry/models.py)
- Provider diffusers: [`unified_diffusion/providers/diffusers_provider.py`](/Users/robinsongarcia/projects/unified_diffusion/unified_diffusion/providers/diffusers_provider.py)
- Provider FLUX: [`unified_diffusion/providers/flux_provider.py`](/Users/robinsongarcia/projects/unified_diffusion/unified_diffusion/providers/flux_provider.py)

## Taxonomia recomendada

Ao avaliar uma nova entrada de catálogo, classifique cada item nestas dimensões:

### 1. Base model family

Exemplos:

- `stable-diffusion`
- `stable-diffusion-xl`
- `stable-diffusion-3`
- `pixart`
- `flux`
- `wan`
- `cogvideox`

Essa dimensão responde: “de que família técnica esse modelo vem?”

### 2. Base model variant

Exemplos:

- `base`
- `refiner`
- `lightning`
- `hyper`
- `sigma`
- `alpha`
- `dev`
- `schnell`

Essa dimensão responde: “qual subvariante da família é essa?”

### 3. Modality

Exemplos:

- `image`
- `video`
- `multi`

Essa dimensão responde: “o artefato final é imagem, vídeo ou multimodal?”

### 4. Generation mode

Exemplos:

- `t2i`
- `i2i`
- `t2v`
- `i2v`
- `upscale`

Essa dimensão responde: “que tipo de entrada/saída o runtime precisa?”

### 5. Artifact type

Exemplos:

- `checkpoint`
- `lora`
- `vae`
- `controlnet`
- `embedding`
- `upscaler`

Essa dimensão responde: “o item é um modelo completo ou um componente auxiliar?”

### 6. File format

Exemplos:

- `diffusers`
- `safetensors`
- `pickle`
- `onnx`
- `coreml`
- `gguf`

Essa dimensão responde: “como o artefato está empacotado?”

### 7. Status flags

Exemplos:

- `featured`
- `early-access`
- `on-site-generation`

Essa dimensão responde: “quais badges editoriais ou flags de catálogo se aplicam?”

## Regra de ouro

Não transforme tudo em flag de geração.

Várias coisas da UI devem ser tratadas como:

- alias
- metadado de catálogo
- capability
- filtro de listagem

e não como parâmetro obrigatório de `run()`.

## O que deve virar flag real e o que deve virar metadado

### Deve influenciar runtime diretamente

- `provider`
- `pipeline_type`
- `modality`
- `generation_mode`
- `artifact_type`
- `revision`
- `device`
- `dtype`

### Deve ser metadado ou filtro de catálogo

- `statusFlags`
- `checkpointType`
- `featured`
- `madeOnSite`
- `earlyAccess`
- `fileFormat` quando não altera o loader real
- `other`

## Como mapear listas de UI para a estrutura atual

### Base model

Na estrutura atual, quase tudo em “Base model” precisa virar:

- `model="<tag_canônica>"`

e, opcionalmente, metadados no registry.

Exemplos diretos hoje:

- `SDXL 1.0 -> model="sdxl.base"`
- `SD 2.1 -> model="sd21.base"`
- `PixArt α -> model="pixart.alpha"`
- `PixArt Σ -> model="pixart.sigma"`
- `FLUX.1 dev -> model="flux.1-dev"`

### Model types

Hoje não há suporte estrutural para quase tudo fora de `checkpoint`.

Mapeamento recomendado:

- `Checkpoint -> artifact_type="checkpoint"`
- `LoRA -> artifact_type="lora"`
- `VAE -> artifact_type="vae"`
- `ControlNet -> artifact_type="controlnet"`

Mas isso ainda não existe no runtime atual.

### Checkpoint type

Não deve virar flag de geração.

Mapeamento recomendado:

- `Trained -> checkpoint_type="trained"`
- `Merge -> checkpoint_type="merge"`

Isso é filtro de catálogo.

### File format

Hoje o runtime suporta de verdade:

- `Diffusers`

e prefere `safetensors` quando o pipeline `from_pretrained()` aceita.

Mapeamento recomendado:

- `Diffusers -> file_formats=["diffusers"]`
- `SafeTensor -> file_formats=["safetensors"]`
- `PickleTensor -> file_formats=["pickle"]`

Mas `fileFormat` não deveria ser o eixo principal do request público.

## Como pesquisar um novo modelo

Checklist mínimo:

1. encontre o `repo_id` exato no Hugging Face
2. leia o model card
3. confirme a licença e se há gating
4. confirme a modalidade real do modelo
5. confirme o modo de geração real
6. confirme se o modelo é checkpoint completo ou artefato auxiliar
7. confirme se ele funciona com `diffusers` ou exige outro backend
8. só então decida `provider` e `pipeline_type`

## Como escolher o pipeline_type

Atualmente, os valores úteis são:

- `stable-diffusion`
- `stable-diffusion-xl`
- `stable-diffusion-3`
- `pixart-alpha`
- `pixart-sigma`
- `flux`

Heurística:

- SD 1.x / 2.x -> `stable-diffusion`
- SDXL e derivados -> `stable-diffusion-xl`
- SD3 -> `stable-diffusion-3`
- PixArt Alpha -> `pixart-alpha`
- PixArt Sigma -> `pixart-sigma`
- FLUX -> `flux`

Se o modelo for vídeo, esse mapeamento atual não basta.

## Quando basta registry externo

Use só registry externo quando:

- o modelo continua sendo `image + t2i + checkpoint`
- o backend continua sendo `diffusers` ou `flux`
- o pipeline é compatível com a lógica atual
- você só quer nova tag, novo repo e talvez alias

Exemplo:

```json
{
  "my.new.model": {
    "provider": "diffusers",
    "source": "org/model-repo",
    "pipeline_type": "stable-diffusion-xl",
    "default_revision": "main",
    "license_hint": "Check upstream license.",
    "notes": "Internal experiment."
  }
}
```

### Exemplo com diretório local

```json
{
  "local.sdxl.dir": {
    "provider": "diffusers",
    "source": "~/models/civitai/local-sdxl-dir",
    "pipeline_type": "stable-diffusion-xl"
  }
}
```

### Exemplo com arquivo `.safetensors`

```json
{
  "local.sdxl.file": {
    "provider": "diffusers",
    "source": "~/models/civitai/local-sdxl/model.safetensors",
    "pipeline_type": "stable-diffusion-xl"
  }
}
```

Regra prática:

- se você tiver um diretório pronto para `diffusers`, prefira o diretório
- se você tiver só um `.safetensors`, vale testar, mas isso depende de suporte a `from_single_file()`

## Quando precisa mudar o registry embutido

Adicione ao registry embutido quando:

- o modelo será suportado oficialmente pelo package
- a tag deve aparecer em `list_models()` por padrão
- você quer documentação nativa em [`MODELS.md`](/Users/robinsongarcia/projects/unified_diffusion/docs/MODELS.md)

## Quando precisa pequena modificação

Pequena modificação significa:

- alias/sinônimo
- novo `pipeline_type`
- metadados extras no registry
- regra simples de compatibilidade
- filtro adicional em `list_models()`

Exemplos:

- `SDXL Lightning`
- `SDXL Hyper`
- `SD 1.5`
- `SD 1.5 LCM`
- `Pony`
- `Illustrious`
- `Anima`

## Quando precisa refactor maior

Refactor maior é necessário quando o item implica:

- `video`
- `t2v`
- `i2v`
- `LoRA`
- `ControlNet`
- `Embedding`
- `Upscaler`
- `Workflows`
- `Detection`
- `ONNX`
- `Core ML`
- `GGUF`

Exemplos típicos:

- `CogVideoX`
- `Wan Video`
- `Hunyuan Video`
- `LTXV`
- `Mochi`

## Modelo de dados alvo de baixo impacto

Se você quiser evoluir a taxonomia sem quebrar compatibilidade, o próximo passo consistente é expandir `ModelSpec` para algo assim:

```python
@dataclass(frozen=True, slots=True)
class ModelSpec:
    canonical_id: str
    provider: str
    source: str
    pipeline_type: str
    default_revision: str | None = None
    license_hint: str | None = None
    notes: str | None = None

    base_model_family: str | None = None
    base_model_variant: str | None = None
    modality: str = "image"
    generation_mode: str = "t2i"
    artifact_type: str = "checkpoint"
    checkpoint_type: str | None = None
    file_formats: tuple[str, ...] = ("diffusers",)
    status_flags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
```

## Fluxo recomendado para adicionar um novo model

1. Classifique o item nas 7 dimensões acima.
2. Verifique se ele ainda cabe em `image + t2i + checkpoint`.
3. Se sim, tente começar por registry externo.
4. Se precisar de suporte oficial, mova para o registry embutido.
5. Se exigir modalidade/artifact type diferente, abra uma mudança estrutural antes de registrar como “suportado”.

## Critério de aceitação para “suportado”

Só considere um modelo “suportado” quando:

- o repo é localizável
- o pipeline certo existe
- o provider consegue baixar
- o provider consegue carregar
- o provider consegue gerar
- a licença e gating foram ao menos documentados

## Não fazer

- não misture badge editorial com parâmetro de inferência
- não trate `Other` como valor canônico de request
- não registre vídeo como se fosse image t2i
- não registre LoRA como se fosse checkpoint completo
- não assuma equivalência sem model card e pipeline compatível
