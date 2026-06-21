# Changelog

All notable changes to this project are documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/);
each section corresponds to a git version tag (which is also the release
published to PyPI). Entries are commit subjects and PR titles, verbatim.

## [0.1.20] - 2026-06-16

- Use get_embedding_dimension (drop sentence-transformers FutureWarning) ([#33](https://github.com/thorwhalen/ef/pull/33))

## [0.1.19] - 2026-06-11

### Added

- feat(evaluation): publish RETRIEVAL_METRICS as the public metric registry ([#32](https://github.com/thorwhalen/ef/pull/32))

## [0.1.18] - 2026-05-27

- ci: refresh stub for new permissions block
- ci: refresh stub for new permissions block
- ci: switch to wads reusable workflow stub

## [0.1.17] - 2026-05-22

- BYOK: thread a per-call embedder_api_key into create_corpus ([#31](https://github.com/thorwhalen/ef/pull/31))

## [0.1.16] - 2026-05-21

- EfService: add per-instance default_embedder hook ([#29](https://github.com/thorwhalen/ef/pull/29))

## [0.1.15] - 2026-05-21

- Adapt to vd 0.2.0: drop the dummy-embedder workaround ([#27](https://github.com/thorwhalen/ef/pull/27))

## [0.1.14] - 2026-05-21

- Phase 2b: explore() orchestrator + EfService.explore_corpus ([#26](https://github.com/thorwhalen/ef/pull/26))

## [0.1.13] - 2026-05-21

- Phase 2a: ef/service.py — EfService stateless-bridge facade ([#24](https://github.com/thorwhalen/ef/pull/24))

## [0.1.12] - 2026-05-21

- Add Cohere / Voyage / Gemini embedder adapters ([#21](https://github.com/thorwhalen/ef/pull/21))

## [0.1.11] - 2026-05-20

- ef Phase 8: demote the visualization pipeline to layer L5 "explore" ([#19](https://github.com/thorwhalen/ef/pull/19))

## [0.1.10] - 2026-05-20

- ef Phase 7: RAG-plug-in retrieve() + evaluation hookpoints ([#17](https://github.com/thorwhalen/ef/pull/17))

## [0.1.9] - 2026-05-20

- ef Phase 6: refresh (explicit + auto) + the four staleness diagnostics ([#15](https://github.com/thorwhalen/ef/pull/15))

## [0.1.8] - 2026-05-20

- Add HashingEmbedder — the dependency-free default embedder ([#13](https://github.com/thorwhalen/ef/pull/13))
- Phase 5: "ready search" + one-shot ingest + the config layer ([#11](https://github.com/thorwhalen/ef/pull/11))

## [0.1.7] - 2026-05-20

- ef Phase 4: the ArtifactGraph core — a content-addressed producer graph ([#9](https://github.com/thorwhalen/ef/pull/9))

## [0.1.6] - 2026-05-20

- Phase 3: the Corpus abstraction + ChangeDetecting wrapper ([#7](https://github.com/thorwhalen/ef/pull/7))

## [0.1.5] - 2026-05-20

- ef Phase 2: the Segmenter facade — data model, protocol, default splitter ([#5](https://github.com/thorwhalen/ef/pull/5))

## [0.1.4] - 2026-05-20

- ef Phase 1: the Embedder facade — protocol, wrappers, adapters ([#3](https://github.com/thorwhalen/ef/pull/3))

## [0.1.3] - 2026-05-20

- Add design notes, use-case catalogue, and ef-architecture dev skill ([#1](https://github.com/thorwhalen/ef/pull/1))

## [0.1.2] - 2026-05-19

- Migrate to wads uv-based CI
- Add initial Jupyter notebook with imports for tonal and i2.castgraph
- improvements
- Refactor code examples in documentation to suppress output; update import paths for clusterers; enhance CI workflow with Python 3.10 and 3.12 support; add .gitignore for build artifacts; create initial Jupyter notebook demo; establish pytest configuration for testing.
- license
- 0.1.1:
- Initial commit
