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

Model and runtime revisions, integrity hashes, installation paths and weight-license boundaries are documented in [Local OpenJev deployment](../guides/local-openjev.md). Model weights and executables are downloaded separately and are not committed.

## Python dependencies

Package versions and platform markers are pinned in `uv.lock` and `requirements.lock.txt`. Each dependency retains its own license. The project does not redistribute API credentials or hosted model weights.
