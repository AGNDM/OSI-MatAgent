# Setup and packaging maintenance

Run commands from the instrument package root inside a compute allocation.

```bash
python3.12 tools/setup.py
# Optional model-inference environment; weights remain separately staged.
python3.12 tools/setup.py --model
.venv/bin/python tools/release.py check --output output/release-check.json
```

Setup verifies existing environments against exact dependency locks and never
upgrades or replaces them. Use `--environment /absolute/fresh/path` for another
destination. Package installation requires an accessible package index or mirror.

For documentation or packaging changes, refresh the file inventory:

```bash
.venv/bin/python tools/release.py refresh --output output/release-refresh.json
```

Refresh authenticates the previous inventory and refuses changes to questions,
answers, splits, case specifications, core graders or vendored runtime. It updates
packaging inventories and validates the resulting copy. Failed validation rolls
back the inventories. Changed scientific content requires a separately reviewed
release. Keep generated environments, caches and outputs out of distributions.
Historical validation receipts retain their original identities.
