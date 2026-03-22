# Model Guide

Este arquivo resume os modelos atualmente registrados no `unified_diffusion`, com foco em uso prático.

## Como ler este guia

- `tag`: valor de `GenerateRequest.model`
- `status`: quão pronto o modelo está no package hoje
- `melhor para`: onde o modelo tende a render melhor
- `prompt strategy`: estrutura recomendada de prompt
- `meta-prompt`: prompt para pedir a outro LLM que escreva um prompt forte para esse modelo

## Modelos customizados

Se você precisar trabalhar com pesos adicionais fora do registry embutido, use um registry externo via `UNIFIED_DIFFUSION_REGISTRY_PATH`.

```bash
export UNIFIED_DIFFUSION_REGISTRY_PATH=$PWD/custom-models.json
```

Na CLI, `udiff` também detecta automaticamente `custom-models.json` no diretório atual quando esse arquivo existir.

Formato mínimo:

```json
{
  "research.custom.one": {
    "provider": "diffusers",
    "source": "org-or-user/model-one",
    "pipeline_type": "stable-diffusion-xl"
  }
}
```

### Como pesquisar modelos

Checklist prático antes de adicionar uma entrada:

1. encontre o `repo_id` correto no Hugging Face
2. leia o model card
3. confirme se há gating/licença
4. confirme que o modelo é compatível com `diffusers`
5. escolha o `pipeline_type` correto

Mapeamento rápido de `pipeline_type`:

- `stable-diffusion`: família SD 1.x / 2.x
- `stable-diffusion-xl`: SDXL e derivados compatíveis
- `stable-diffusion-3`: SD3
- `pixart-alpha`: PixArt Alpha
- `pixart-sigma`: PixArt Sigma
- `flux`: FLUX

### O que significa cada campo

- `canonical_id`:
  - não aparece quando você usa o formato JSON por objeto
  - é o nome da tag que você vai usar em `GenerateRequest(model="...")`
- `provider`:
  - backend interno usado pelo package
  - hoje: `diffusers` ou `flux`
- `source`:
  - origem que o provider consegue usar
  - pode ser `repo_id`, diretório local ou arquivo `.safetensors`
- `pipeline_type`:
  - classe lógica do pipeline; ajuda o package a escolher a família certa
- `default_revision`:
  - opcional
  - branch/tag/revision padrão
- `license_hint`:
  - opcional
  - aparece em log e documentação, mas não faz enforcement automático
- `notes`:
  - opcional
  - observações livres sobre compatibilidade, gating ou finalidade

## sdxl.base

- Nome: Stable Diffusion XL Base 1.0
- Tag: `sdxl.base`
- Status: pronto para uso
- Melhor para: ilustração, concept art, stills cinematográficos, publicidade simples, cenas gerais de alta resolução
- O que faz: é o modelo base do SDXL e pode ser usado sozinho para gerar imagens 1024px a partir de texto. Ele tende a ser versátil, fácil de controlar e um bom ponto de partida para a maioria dos casos.
- Potencialidades:
  - boa qualidade geral para retratos, objetos, cenários e visual publicitário
  - boa disponibilidade de exemplos e ecossistema
  - funciona bem com prompts descritivos em linguagem natural
- Limitações práticas:
  - pode falhar em texto legível dentro da imagem
  - composicionalidade complexa ainda pode degradar
  - em Apple Silicon, prefira o fluxo atual do package, que força `fp32` em `mps` para evitar imagens pretas
- Prompt strategy:
  - sujeito principal
  - contexto/ambiente
  - lente/estilo visual
  - iluminação
  - materiais/texturas
  - enquadramento
  - quality cues no fim
- Exemplo de prompt:
  - `a red cube on a white table, studio lighting, clean commercial product photography, soft shadows, 50mm lens, minimal background, high detail`
- Meta-prompt para gerar prompt:
  - `Escreva um prompt para Stable Diffusion XL Base 1.0 focado em text-to-image. Estruture em: sujeito principal, ambiente, composição, iluminação, câmera/lente, textura/material, estilo visual e detalhes de qualidade. Evite texto dentro da cena. Entregue uma única linha, em inglês, pronta para uso.`
- Referência externa:
  - https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

## sdxl.refiner

- Nome: Stable Diffusion XL Refiner 1.0
- Tag: `sdxl.refiner`
- Status: registrado, mas uso prático limitado na API mínima atual
- Melhor para: refino dos passos finais de uma imagem ou latente já gerada pelo SDXL base
- O que faz: o refiner foi pensado para a segunda etapa do SDXL, melhorando detalhes finais e denoising tardio. Ele não é a melhor escolha como primeiro modelo para text-to-image puro dentro da API atual, porque o uso ideal envolve pipeline em duas etapas.
- Potencialidades:
  - melhora detalhe fino
  - pode elevar acabamento visual quando usado junto do base
- Limitações práticas:
  - não é o fluxo principal desta API mínima hoje
  - para melhor resultado, o ideal é pipeline base + refiner em dois estágios
- Prompt strategy:
  - reutilize o mesmo prompt usado no `sdxl.base`
  - trate o modelo como refinador, não como gerador inicial
- Meta-prompt para gerar prompt:
  - `Escreva um prompt em inglês para SDXL Base + Refiner. O prompt deve ser rico em detalhes visuais, coerente e reaproveitável nas duas etapas. Produza uma única linha pronta para uso.`
- Referência externa:
  - https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0

## sd21.base

- Nome: Stable Diffusion 2.1 Base
- Tag: `sd21.base`
- Status: pronto para uso
- Melhor para: geração geral mais leve que SDXL, experimentação rápida e prompts simples
- O que faz: é uma geração anterior da família Stable Diffusion, mas ainda bastante útil para text-to-image geral, prototipagem e cenários onde você quer um modelo mais conhecido e relativamente simples de operar.
- Potencialidades:
  - mais leve que SDXL em vários cenários
  - bom para iteração rápida
  - ecossistema amplo e comportamento previsível
- Limitações práticas:
  - qualidade geral inferior a SDXL e FLUX
  - menos capacidade de detalhes finos e composição complexa
- Prompt strategy:
  - sujeito claro
  - ambiente
  - estilo
  - iluminação
  - acabamento
- Exemplo de prompt:
  - `a cozy reading corner by a large window, warm afternoon light, realistic interior photography, soft shadows, clean composition`
- Meta-prompt para gerar prompt:
  - `Escreva um prompt em inglês para Stable Diffusion 2.1 Base, priorizando clareza e composição simples. Inclua sujeito, ambiente, iluminação, estilo e acabamento visual. Retorne uma única linha pronta para uso.`
- Referência externa:
  - https://huggingface.co/stabilityai/stable-diffusion-2-1-base

## playground.v25

- Nome: Playground v2.5 1024px Aesthetic
- Tag: `playground.v25`
- Status: pronto quando o modelo estiver acessível no Hugging Face
- Melhor para: imagens esteticamente fortes, direção de arte, editorial, lifestyle e concept visuals
- O que faz: é um modelo orientado a qualidade estética e direção visual, com foco em resultados atraentes em 1024px.
- Potencialidades:
  - estética forte já no modelo base
  - bom para visuais publicitários e imagens com apelo artístico
  - alternativa criativa ao SDXL padrão
- Limitações práticas:
  - pode privilegiar “look” sobre literalidade do prompt
  - compatibilidade e gating dependem do modelo publicado no Hugging Face
- Prompt strategy:
  - sujeito
  - direção de arte
  - paleta ou mood
  - luz
  - lente/composição
  - acabamento final
- Exemplo de prompt:
  - `an editorial fashion portrait in a brutalist concrete interior, muted earth tones, sculptural side lighting, medium format photography, refined luxury aesthetic`
- Meta-prompt para gerar prompt:
  - `Crie um prompt em inglês para Playground v2.5 com foco em direção de arte e apelo estético. Inclua sujeito, mood, paleta, iluminação, composição e acabamento visual. Retorne uma única linha pronta para inferência.`
- Referência externa:
  - https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic

## pixart.alpha

- Nome: PixArt-α XL 2 1024 MS
- Tag: `pixart.alpha`
- Status: pronto quando o `diffusers` expõe `PixArtAlphaPipeline`
- Melhor para: cenas 1024px em estilo detalhado com abordagem transformer-first
- O que faz: PixArt-α é um modelo text-to-image baseado em blocos transformer para latent diffusion. A proposta central é gerar imagens 1024px diretamente em uma única etapa de amostragem.
- Potencialidades:
  - boa aderência a prompts descritivos
  - arquitetura transformer dedicada a text-to-image
  - interessante para comparar contra SDXL em cenas semânticas mais longas
- Limitações práticas:
  - compatibilidade depende da versão instalada de `diffusers`
  - o comportamento pode variar mais entre versões do stack
- Prompt strategy:
  - sujeito
  - ação ou pose
  - ambiente
  - estilo visual
  - detalhes materiais e de luz
  - nível de acabamento
- Exemplo de prompt:
  - `a futuristic tea house in a bamboo forest, morning fog, cinematic composition, volumetric light, intricate wood textures, serene atmosphere, highly detailed`
- Meta-prompt para gerar prompt:
  - `Crie um prompt em inglês para PixArt-alpha, focado em descrição semântica clara e rica. Inclua sujeito, ação, ambiente, estilo, iluminação, materiais e acabamento visual. Retorne uma única linha pronta para inferência.`
- Referência externa:
  - https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS

## pixart.sigma

- Nome: PixArt-Σ XL 2 1024 MS
- Tag: `pixart.sigma`
- Status: pronto quando o `diffusers` expõe `PixArtSigmaPipeline`
- Melhor para: composições maiores, cenas detalhadas e cenários em resoluções mais altas
- O que faz: PixArt-Σ é a evolução do PixArt com foco em geração 1024px, 2K e 4K, ainda baseada em transformer latent diffusion.
- Potencialidades:
  - melhor vocação para resoluções altas
  - bom para cenas arquitetônicas, ambientes amplos e composições detalhadas
- Limitações práticas:
  - mais pesado
  - depende da versão do `diffusers`
- Prompt strategy:
  - cena principal
  - escala espacial
  - câmera e perspectiva
  - luz
  - materiais
  - atmosfera
  - acabamento
- Meta-prompt para gerar prompt:
  - `Escreva um prompt em inglês para PixArt-Sigma, pensando em uma cena detalhada de alta resolução. Inclua escala espacial, perspectiva de câmera, iluminação, materiais, atmosfera e acabamento visual. Entregue uma única linha pronta para uso.`
- Referência externa:
  - https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS

## sd3.medium

- Nome: Stable Diffusion 3 Medium
- Tag: `sd3.medium`
- Status: experimental, condicionado a acesso/licença do modelo no Hugging Face
- Melhor para: prompt following mais forte, tipografia melhor e prompts complexos
- O que faz: SD3 Medium é um modelo MMDiT com foco em melhor qualidade de imagem, melhor entendimento de prompt complexo, tipografia superior e maior eficiência de recursos.
- Potencialidades:
  - melhor compreensão semântica
  - melhor texto/typography do que gerações anteriores
  - bom para prompts mais longos e estruturados
- Limitações práticas:
  - requer aceitar os termos do modelo no Hugging Face
  - licenciamento é mais restritivo
  - compatibilidade depende do `diffusers`
- Prompt strategy:
  - descrição principal clara
  - hierarquia explícita de elementos
  - texto desejado, se houver
  - estilo visual e intenção
  - constraints negativas
- Meta-prompt para gerar prompt:
  - `Escreva um prompt em inglês para Stable Diffusion 3 Medium com forte obediência semântica. Estruture prioridade de elementos, estilo visual, composição, iluminação e, se houver, texto literal que precisa aparecer na cena. Entregue uma única linha pronta para uso.`
- Referência externa:
  - https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers

## flux.1-dev

- Nome: FLUX.1 [dev]
- Tag: `flux.1-dev`
- Status: experimental, mas já suportado pelo provider quando o ambiente tem `FluxPipeline`
- Melhor para: prompt following forte, imagens de alta qualidade e testes comparativos modernos
- O que faz: FLUX.1 [dev] é um rectified flow transformer de 12B parâmetros para geração de imagens a partir de texto.
- Potencialidades:
  - qualidade de saída muito forte
  - bom prompt following
  - bom candidato para workflows mais modernos
- Limitações práticas:
  - requer aceitar os termos do modelo no Hugging Face
  - é pesado e pode exigir mais memória/tempo
  - o desempenho local depende muito do hardware
- Prompt strategy:
  - declaração curta do sujeito
  - atributos visuais fortes
  - ambiente
  - cinematografia ou direção de arte
  - acabamento final
- Exemplo de prompt:
  - `a minimalist chair in a white studio, premium product photography, sculptural design, soft softbox lighting, subtle floor reflection, ultra clean composition, high-end editorial look`
- Meta-prompt para gerar prompt:
  - `Crie um prompt em inglês para FLUX.1-dev com forte clareza visual e alto poder de direção artística. Inclua sujeito, materiais, iluminação, composição, contexto e acabamento visual. Entregue uma única linha pronta para text-to-image.`
- Referência externa:
  - https://huggingface.co/black-forest-labs/FLUX.1-dev

## Resumo rápido

- Uso mais direto hoje: `sdxl.base`
- Opção mais leve para iteração: `sd21.base`
- Opção com estética forte: `playground.v25`
- Melhor candidato moderno se você tiver acesso e hardware: `flux.1-dev`
- Melhor para prompt following avançado, com gating/licença: `sd3.medium`
- Alternativas transformer com foco em alta resolução: `pixart.alpha`, `pixart.sigma`
