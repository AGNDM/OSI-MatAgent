# Local OpenJev Flash 9B Q4 Router

The router uses `openjev/OpenJev-Flash-9B-GGUF`, quantization `Q4_K_M`, through llama.cpp. It is a different model/runtime integration from the original proposal's OpenJev 4B reference. General specialists use a separate backend.

## Pinned artifacts

- Model repository: <https://huggingface.co/openjev/OpenJev-Flash-9B-GGUF>
- Model revision: `a3acd96966b06c2f161209f888b66235029f9ac1`
- File: `OpenJev-Flash-9B-Q4_K_M.gguf`
- Size: 5,629,109,120 bytes, approximately 5.24 GiB
- SHA-256: `6209cf51aec005b00cb80fdfef1b961adde57d5a8f181a377bd3882a4668b91d`
- Runtime: official llama.cpp `b11147`, commit `fee39dd92`, Linux x86-64 Vulkan binaries
- Weight license: CC BY-NC 4.0; retain the downloaded license and notices

## Install and serve

```bash
conda activate osi-hack
python tools/install_openjev_runtime.py
python tools/fetch_openjev.py
python tools/serve_openjev.py
```

Weights and executables live under `~/.local/share/osi-hack/`, outside Git. On the development machine, executables/shared libraries use the Linux home filesystem rather than the project data filesystem.

The default service listens on `127.0.0.1:8792` with a 2,048-token context, one slot, eight CPU threads and a bounded prompt cache. Vulkan uses the integrated GPU. For CPU-only execution or shutdown:

```bash
python tools/serve_openjev.py --cpu
python tools/serve_openjev.py --stop
```

The helper installs no system startup service. Avoid starting a second copy when the intended service is already healthy. The optional `--xet` download path requires separately installed Hugging Face tooling; the default download path uses the standard library.

## Workflow integration

```bash
# Real router, explicitly scripted specialists.
python -m bench.device_tasks --router openjev --device mt_xpr204e \
  --output results/my-router-check

# Real router and separately configured real DeepSeek specialists.
python -m bench.device_tasks --backend deepseek --router openjev \
  --device mt_xpr204e --output results/my-combined-check
```

The second command sends cloud requests and uses the shared device-debugging token ledger. See [Device workflow](device-workflow.md). Routing failures use an explicitly logged rules fallback; a run with fallback is not evidence of uninterrupted model routing.

## Probability extraction

`app/local_decision.py` applies the model's state/question/letter-options template, calls the local completion endpoint and reads one-token candidate-letter probabilities. All candidate letters must be present in the probability readout. Their probabilities are normalized within the candidate set; confidence is not fabricated from generated prose or filled with zeros when missing.

The adapter uses greedy generation while reading the runtime's T=1 log probabilities. In the pinned runtime, post-sampling probabilities were observed to include vocabulary outside the grammar; the adapter does not misrepresent that as an already candidate-constrained distribution. The resulting candidate probabilities are not calibrated on laboratory tasks and do not implement a claimed official calibration service.

## Recorded deployment checks

Both CPU and Radeon 780M Vulkan execution were tested. Historical fixed probes reported Vulkan 6/6 correct decisions over two rounds, median 1.946 seconds, and CPU 3/3, median 4.564 seconds. These tiny probes do not establish general accuracy or a full-task speed advantage. Historical startup measurements included integrity checks and potentially warm filesystem/driver caches.

For the device expansion, three scripted-specialist/OpenJev runs and two real-DeepSeek/OpenJev runs passed without routing fallback. See [Evaluation summary](../reports/evaluation-summary.md). Detailed deployment measurements and initial failed probability-adapter attempts remain in the ignored local results directory.
