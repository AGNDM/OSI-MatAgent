# Third-Party Components and Provenance

## ChemLab-Assets

Source repository: <https://github.com/mbzuai-nlp/ChemLab-Assets>

Dataset: <https://huggingface.co/datasets/MBZUAI/ChemLab-Assets>

- Code revision: `123cf78593469871e41b8dbd63a11dfb7c5ca251`
- Data revision: `4ca4a42dab2ca0a884e449d202e9ad96625d5648`
- Upstream code license: [MIT](../../vendor/chemlab/LICENSE)
- Upstream asset/data terms: [LICENSE-ASSETS.md](../../vendor/chemlab/LICENSE-ASSETS.md)
- File/source-specific attribution and additional terms: [NOTICE.md](../../vendor/chemlab/NOTICE.md)

The three manifests under `vendor/chemlab/` record the selected files and their SHA-256 hashes. Native code and data are restored unmodified with `python tools/fetch_device_runtime.py --download`; their downloaded contents are ignored by Git. Upstream documentation and multilingual data remain in their original form in that local cache. No repository-wide license is implied by the upstream code license.

## OpenJev and llama.cpp

The optional local backend uses `openjev/OpenJev-Flash-9B-GGUF` at revision `a3acd96966b06c2f161209f888b66235029f9ac1`, file `OpenJev-Flash-9B-Q4_K_M.gguf`, with llama.cpp `b11147` (`fee39dd92`). The weights use CC BY-NC 4.0; retain their downloaded license and notices. Installation helpers are `tools/install_openjev_runtime.py`, `tools/fetch_openjev.py` and `tools/serve_openjev.py`. Artifacts live under `~/.local/share/osi-hack/`, outside Git. Hosted TypeSafe Jev 1.14 is a separate service, not this local model.

## Python dependencies

Package versions and platform markers are pinned in `uv.lock` and `requirements.lock.txt`. Each dependency retains its own license. The project does not redistribute API credentials or hosted model weights.
