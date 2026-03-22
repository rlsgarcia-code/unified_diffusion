# Release Process

Guia curto para versionamento e release do `unified_diffusion`.

## Objetivo

Manter um fluxo simples e previsível para evolução do package.

## Antes de um release

1. atualizar código e documentação necessária
2. rodar validações locais:

```bash
uv run pytest
uv run ruff check .
npm run smoke:api:local
npm run smoke:api:docker
```

3. revisar mudanças em:

- [`CHANGELOG.md`](/Users/robinsongarcia/projects/unified_diffusion/CHANGELOG.md)
- [`pyproject.toml`](/Users/robinsongarcia/projects/unified_diffusion/pyproject.toml)

## Versionamento

Use semver simples:

- patch: correções e docs sem breaking change
- minor: features novas compatíveis
- major: breaking change intencional

Arquivos a atualizar:

- [`pyproject.toml`](/Users/robinsongarcia/projects/unified_diffusion/pyproject.toml)
- [`CHANGELOG.md`](/Users/robinsongarcia/projects/unified_diffusion/CHANGELOG.md)

## Checklist mínimo de release

1. atualizar `version` em `pyproject.toml`
2. adicionar seção nova em `CHANGELOG.md`
3. validar:
   - testes
   - lint
   - docs críticas
4. criar commit de release
5. criar tag correspondente, se aplicável
6. enviar a tag `v*` para disparar o workflow de release automatizado

## Quando atualizar o changelog

Atualize quando houver:

- nova capability pública
- alteração de comportamento
- mudança relevante de documentação operacional
- novo endpoint de API
- mudança em registry/cache/CLI

## Quando NÃO incrementar major

Em geral não precisa major para:

- docs
- novos modelos adicionados ao registry
- novos comandos CLI compatíveis
- novos endpoints compatíveis

## Automação atual

O repositório já possui:

- CI em [ci.yml](/Users/robinsongarcia/projects/unified_diffusion/.github/workflows/ci.yml)
- release por tag em [release.yml](/Users/robinsongarcia/projects/unified_diffusion/.github/workflows/release.yml)

O workflow de release:

- roda testes
- roda lint
- executa smoke local da API
- executa smoke Docker da API
- gera `sdist` e `wheel`
- publica os artefatos em um GitHub Release

## Observação

Se o package passar a ser publicado em índice de pacotes depois, o próximo passo natural é adicionar publicação no PyPI ao workflow de release.
