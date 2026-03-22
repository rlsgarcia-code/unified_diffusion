# Civitai Guide

Este documento explica o fluxo manual pelo site do Civitai para descobrir modelos, filtrar os que fazem sentido para o `unified_diffusion`, baixar o artefato certo e transformar isso em uma entrada válida do registry.

O foco aqui não é a API pública do Civitai. O foco é:

1. filtrar no site
2. decidir se o modelo cabe na estrutura atual do package
3. baixar o artefato certo
4. salvar no lugar certo
5. preencher o JSON do registry
6. validar até o modelo funcionar

## O que o package suporta hoje

O runtime atual foi desenhado para este caso principal:

- modalidade: `image`
- modo de geração: `text-to-image`
- artefato: `checkpoint` ou pipeline completo
- backend: `diffusers` ou `flux`
- formato estrutural utilizável pelo loader atual

Na prática, o package hoje suporta bem:

- SD 1.x / 2.x style checkpoints, quando houver forma compatível de carregar
- SDXL e derivados
- PixArt Alpha / Sigma
- SD3
- FLUX

Ele não suporta hoje, sem refactor:

- vídeo
- `t2v`
- `i2v`
- LoRA
- LyCORIS
- DoRA
- ControlNet
- Embedding / Textual Inversion
- Hypernetwork
- Upscaler como artefato separado
- VAE como artefato separado
- workflows
- detection
- ONNX / Core ML / GGUF como formato nativo de provider

## O que procurar no site do Civitai

Use a página de modelos do Civitai e pense nela em 4 camadas:

1. tipo do artefato
2. família/base model
3. formato/arquitetura prática
4. compatibilidade com o runtime do package

## Como filtrar no site do Civitai

### Regra principal

Para o `unified_diffusion`, o filtro mais seguro é começar por:

- `Model types -> Checkpoint`

e só depois refinar por família.

### Categorias da UI que fazem sentido hoje

#### Base model

Essas são categorias que podem fazer sentido para a estrutura atual, desde que o artefato seja realmente um checkpoint compatível:

- `Flux .1 S`
- `Flux .1D`
- `Flux .1 Krea`
- `Flux .1 Kontext`
- `Flux .2 D`
- `Flux .2 Klein 9B`
- `Flux .2 Klein 9B-base`
- `Flux .2 Klein 4B`
- `Flux .2 Klein 4B-base`
- `PixArt α`
- `PixArt Σ`
- `SD 1.4`
- `SD 1.5`
- `SD 1.5 LCM`
- `SD 1.5 Hyper`
- `SD 2.0`
- `SD 2.1`
- `SDXL 1.0`
- `SDXL Lightning`
- `SDXL Hyper`
- `Pony`
- `Pony V7`
- `Illustrious`
- `Anima`
- `Kolors`
- `Lumina`
- `Aura Flow`
- `Chroma`
- `HiDream`
- `NoobAI`
- `Z Image Turbo`
- `Z Image Base`

Essas categorias normalmente exigem verificação extra antes de entrar no package:

- `Hunyuan 1`
- `Qwen`
- `Other`

Essas categorias não fazem sentido para o runtime atual como “novo model suportado”:

- `CogVideoX`
- `Hunyuan Video`
- `LTXV`
- `LTXV2`
- `LTXV 2.3`
- `Mochi`
- `Wan Video 1.3B t2v`
- `Wan Video 14B t2v`
- `Wan Video 14B i2v 480p`
- `Wan Video 14B i2v 720p`
- `Wan Video 2.2 T2V-5B`
- `Wan Video 2.2 I2V-A14B`
- `Wan Video 2.2 T2V-A14B`
- `Wan Video 2.5 T2V`
- `Wan Video 2.5 I2V`

Motivo:

- são vídeo ou implicam modos `t2v` / `i2v`
- a arquitetura atual do package é image `t2i`

#### Model status

No site, categorias como:

- `Early Access`
- `On-site Generation`
- `Made On-site`
- `Featured`

devem ser tratadas como metadado editorial, não como critério principal de compatibilidade técnica.

Elas podem ajudar você a priorizar exploração, mas não dizem se o modelo cabe no package.

#### Model types

Para a estrutura atual, o filtro útil é:

- `Checkpoint`

As demais categorias não entram diretamente no runtime atual:

- `Embedding`
- `Hypernetwork`
- `Aesthetic Gradient`
- `LoRA`
- `LyCORIS`
- `DoRA`
- `Controlnet`
- `Upscaler`
- `Motion`
- `VAE`
- `Poses`
- `Wildcards`
- `Workflows`
- `Detection`
- `Other`

#### Checkpoint type

No site, `Trained` e `Merge` podem ser úteis como sinal de origem do modelo, mas no package atual isso deve ser tratado como observação de catálogo, não como flag de geração.

#### File format

Para o package atual:

- `Diffusers` é o melhor caminho
- `SafeTensor` pode ser útil, mas só se houver forma compatível de carregamento no backend

Os formatos abaixo não têm suporte nativo hoje:

- `PickleTensor`
- `GGUF`
- `Core ML`
- `ONNX`

## Estratégia de filtragem recomendada no site

### Fluxo mais seguro

1. abra a busca de modelos
2. filtre `Model types = Checkpoint`
3. escolha uma família em `Base model`
4. abra a página do modelo
5. confirme se ele é imagem e não vídeo
6. confirme se ele é checkpoint completo e não LoRA/VAE/etc.
7. confirme a licença
8. confirme o formato real disponibilizado
9. confirme se existe caminho real para usar no backend atual

### Ordem de priorização

Se você quer maximizar chance de sucesso no package atual, priorize:

1. `SDXL 1.0`
2. `SD 2.1`
3. `PixArt α`
4. `PixArt Σ`
5. `Flux` variants
6. derivados bem conhecidos de SDXL

## O que baixar

Baixe apenas o artefato principal que corresponda a um modelo base utilizável pelo runtime.

### Baixar quando

- é um `Checkpoint`
- a página deixa claro que é modelo base de geração
- não é só adapter
- não é só VAE
- não é só LoRA
- não é só embedding

### Sobre `.safetensors`

Sim: `.safetensors` normalmente é checkpoint válido, e esse é o formato que mais faz sentido dentro do fluxo atual.

Mas existem dois cenários:

1. diretório completo compatível com `diffusers`
2. arquivo único `.safetensors`

No package atual:

- diretório completo é o caminho mais seguro
- arquivo único `.safetensors` pode funcionar quando a classe de pipeline suporta `from_single_file()`

Regra prática:

- se o autor disponibiliza estrutura pronta para `diffusers`, prefira isso
- se você só tiver o `.safetensors`, ainda vale testar, mas o `pipeline_type` precisa combinar com uma classe que saiba abrir single-file

### Não baixar como “novo model suportado” quando

- o item é `LoRA`
- o item é `ControlNet`
- o item é `Textual Inversion`
- o item é `VAE`
- o item é `Upscaler`
- o item é `Motion`
- o item é vídeo

## Onde salvar

### Se você vai só pesquisar e organizar

Use uma área de staging fora do cache principal:

```text
~/Downloads/civitai/
~/Downloads/civitai/<slug-do-modelo>/
```

ou dentro do projeto:

```text
/Users/robinsongarcia/projects/unified_diffusion/tmp/civitai/
```

Sugestão prática:

```text
~/models/civitai/
~/models/civitai/<slug-do-modelo>/
```

Se você tiver um diretório compatível com `diffusers`:

```text
~/models/civitai/my-model/
  model_index.json
  unet/
  vae/
  text_encoder/
  tokenizer/
  scheduler/
```

Se você tiver um arquivo único:

```text
~/models/civitai/my-model/model.safetensors
```

### Se você quer usar no fluxo normal do package

O package trabalha melhor quando o modelo entra por:

- registry embutido
- registry externo
- e download normal do provider para o cache do package

Ou seja:

- descubra no Civitai
- valide o modelo
- traduza isso para uma entrada do registry
- deixe o provider baixar/carregar pelo fluxo do package

Se o peso for local, “traduzir para o registry” significa apontar `source` para esse path local.

### Quando salvar manualmente no cache do package

Só faça isso se você souber exatamente como o loader espera a estrutura.

Layout atual do cache:

```text
~/.cache/unified-diffusion/
  models/<canonical_id>/<revision>/
```

Mas para o runtime atual isso só funciona bem se o conteúdo salvo ali estiver no formato que `from_pretrained()` espera.

Então a recomendação é:

- não copie um arquivo qualquer do Civitai diretamente para `models/<canonical_id>/<revision>/` achando que o provider vai entender automaticamente

## Como preencher o JSON

O JSON do registry externo deve descrever um modelo que o package consegue resolver.

Exemplo mínimo:

```json
{
  "my.new.model": {
    "provider": "diffusers",
    "source": "org-or-user/model-repo",
    "pipeline_type": "stable-diffusion-xl"
  }
}
```

Campos:

- `provider`
  - hoje: `diffusers` ou `flux`
- `source`
  - identificador que o provider consegue usar
  - pode ser `repo_id`, diretório local ou arquivo `.safetensors`
- `pipeline_type`
  - família lógica do pipeline
- `default_revision`
  - opcional
- `license_hint`
  - opcional
- `notes`
  - opcional

## Como escolher o pipeline_type

Use esta heurística:

- SD 1.x / 2.x -> `stable-diffusion`
- SDXL e derivados -> `stable-diffusion-xl`
- SD3 -> `stable-diffusion-3`
- PixArt α -> `pixart-alpha`
- PixArt Σ -> `pixart-sigma`
- FLUX -> `flux`

## Exemplo de tradução Civitai -> registry

Imagine que você encontrou no Civitai um modelo SDXL derivado e concluiu que ele é checkpoint image `t2i`.

JSON:

```json
{
  "research.illustrated-xl": {
    "provider": "diffusers",
    "source": "org-or-user/illustrated-xl",
    "pipeline_type": "stable-diffusion-xl",
    "default_revision": "main",
    "license_hint": "Verify the upstream terms from the model page.",
    "notes": "Discovered manually on Civitai; validated as image checkpoint."
  }
}
```

Você também pode começar a partir do template:

- [`custom-models.example.json`](/Users/robinsongarcia/projects/unified_diffusion/custom-models.example.json)

Exemplo com peso salvo localmente como arquivo:

```json
{
  "civitai.local.sdxl": {
    "provider": "diffusers",
    "source": "~/models/civitai/my-sdxl/model.safetensors",
    "pipeline_type": "stable-diffusion-xl",
    "license_hint": "Verify the creator terms from the Civitai page.",
    "notes": "Manual local import from Civitai."
  }
}
```

Exemplo com diretório local:

```json
{
  "civitai.local.sdxl.dir": {
    "provider": "diffusers",
    "source": "~/models/civitai/my-sdxl-dir",
    "pipeline_type": "stable-diffusion-xl",
    "license_hint": "Verify the creator terms from the Civitai page.",
    "notes": "Manual local import from Civitai."
  }
}
```

Uso:

```bash
export UNIFIED_DIFFUSION_REGISTRY_PATH=$PWD/custom-models.json
uv run python -c "from unified_diffusion import Diffusion; print(Diffusion().list_models())"
```

## Como validar se o model funciona

### Etapa 1: validar que o registry reconhece a tag

```bash
export UNIFIED_DIFFUSION_REGISTRY_PATH=$PWD/custom-models.json
uv run python - <<'PY'
from unified_diffusion import Diffusion
print(Diffusion().list_models())
PY
```

Você deve ver sua tag nova na lista.

### Etapa 2: validar resolução mínima

```bash
export UNIFIED_DIFFUSION_REGISTRY_PATH=$PWD/custom-models.json
uv run python - <<'PY'
from unified_diffusion import Diffusion, GenerateRequest

engine = Diffusion(cache_dir="~/.cache/unified-diffusion")
result = engine.run(
    GenerateRequest(
        model="research.illustrated-xl",
        prompt="a ceramic teapot on a wooden table, studio light",
        width=512,
        height=512,
        steps=4,
        device="mps",
        seed=1234,
    )
)
result.images[0].save("validation.png")
print(result.model_resolved)
print(result.metadata)
PY
```

### Etapa 3: validar a imagem gerada

Cheque:

- o arquivo foi salvo
- a imagem não está toda preta
- o metadata mostra `provider_used` e `model_resolved` corretos
- se houver fallback de `dtype`, isso aparece em `metadata["warnings"]`

### Etapa 4: validar se o modelo realmente “cabe”

Se ele falhar por:

- formato incompatível
- tipo de artefato errado
- modalidade errada
- pipeline inexistente

então ele não cabe ainda na estrutura atual, mesmo que o download tenha sido possível.

### Sinal de que o `.safetensors` foi entendido corretamente

Você tende a ver sucesso quando:

- o provider não falha na carga
- a classe de pipeline correta é resolvida
- a inferência completa roda
- a imagem final sai válida

Se falhar no load com erro ligado a `from_single_file()`, os motivos mais comuns são:

- `pipeline_type` errado
- arquivo incompatível com a família presumida
- pipeline sem suporte a single-file nesse caso

## Critério de aceite para um modelo vindo do Civitai

Considere “funciona no package” apenas se:

1. a tag aparece em `list_models()`
2. o provider resolve o model
3. o loader abre o pipeline
4. a inferência roda
5. a imagem sai válida

## Não fazer

- não usar `Model status` como critério técnico principal
- não usar `LoRA`, `VAE`, `ControlNet` ou `Embedding` como se fossem `base model`
- não registrar modelo de vídeo como se fosse `image t2i`
- não assumir que todo arquivo baixado do Civitai cabe no loader atual
- não copiar arquivo solto para o cache do package sem saber a estrutura esperada

## Referências usadas

- Civitai wiki home: https://github.com/civitai/civitai/wiki/Home
- Civitai REST API reference: https://github.com/civitai/civitai/wiki/REST-API-Reference
- Civitai how to use models: https://github.com/civitai/civitai/wiki/How-to-use-models
