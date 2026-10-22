"""Pinned local LLM baselines with explicit evidence and probability semantics."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import time

from .io import task_text

SYSTEM = """You operate a simulated laboratory instrument. Use ONLY the supplied public observations, history, instrument contract, and operating notes. Select the best option. Commands being accepted do not establish that a timed operational goal has completed. Request more information when it is actually insufficient, and request maintenance when public controls cannot repair a persistent fault. Option identifiers are arbitrary and shuffled. Do not assume access to evaluator state or fault injection."""


class LocalLLM:
    def __init__(self, model_path="external/models/qwen2.5-3b", device="cuda:0", max_context=30000):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, device
        load_started = time.perf_counter()
        self.path = str(Path(model_path).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                                                         local_files_only=True, attn_implementation="sdpa").to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.max_context = min(max_context, self.model.config.max_position_embeddings - 256)
        self.identity = {"path": model_path, "repository": "Qwen/Qwen2.5-3B-Instruct",
                         "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1", "device": device,
                         "weights_frozen": True, "max_context": self.max_context}
        self.identity["load_wall_time_s"] = time.perf_counter() - load_started

    def inputs(self, task, instruction):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task_text(task) + "\n" + instruction}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        count = inputs.input_ids.shape[-1]
        if count > self.max_context:
            raise ValueError(f"Public evidence is {count} tokens, exceeding {self.max_context}; no evidence was silently truncated")
        return inputs

    def label_scores(self, task):
        ids = [o["option_id"] for o in task["options"]]
        encoded = [self.tokenizer.encode(key, add_special_tokens=False) for key in ids]
        if any(len(tokens) != 1 for tokens in encoded):
            raise ValueError("Single-token log-probability baseline needs single-token option identifiers")
        instruction = "Reply with exactly one option identifier from " + ", ".join(ids) + ". Do not include any other text."
        inputs = self.inputs(task, instruction)
        with self.torch.inference_mode():
            output = self.model(**inputs, use_cache=False)
            logits = output.logits[0, -1, [tokens[0] for tokens in encoded]].float()
            distribution = logits.softmax(-1).cpu().tolist()
        return ids, distribution, inputs.input_ids.shape[-1]

    def predict_logprobs(self, task):
        start = time.perf_counter()
        ids, scores, tokens = self.label_scores(task)
        selected = ids[max(range(len(ids)), key=lambda i: scores[i])]
        return {"selected_option_id": selected, "probabilities": dict(zip(ids, scores)), "status": "ok", "causes_cited": [],
                "input_tokens": tokens, "output_tokens": 0, "api_cost_usd": 0.0,
                "gpu_seconds": time.perf_counter() - start if self.device.startswith("cuda") else None,
                "metadata": {**self.identity, "algorithm": "conditional next-token option softmax", "conditional_on_menu": True}}

    def predict_verbalized(self, task):
        start = time.perf_counter()
        instruction = ('Return ONLY JSON: {"selected_option_id":"A", "confidence":0.0, "causes_cited":["public.slot"]}. '
                       'confidence is your probability that the selected option is correct, between zero and one. Cite public slot or contract precondition identifiers.')
        inputs = self.inputs(task, instruction)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=192, do_sample=False,
                                            pad_token_id=self.tokenizer.eos_token_id)
        suffix = generated[0, inputs.input_ids.shape[-1]:]
        raw = self.tokenizer.decode(suffix, skip_special_tokens=True)
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        accounting = {"input_tokens": inputs.input_ids.shape[-1], "output_tokens": len(suffix),
                      "api_cost_usd": 0.0,
                      "gpu_seconds": time.perf_counter()-start if self.device.startswith("cuda") else 0.0,
                      "metadata": {**self.identity, "algorithm": "verbalized confidence", "raw_completion": raw,
                                   "residual_mass": "uniform over unselected options"}}
        try:
            parsed = json.loads(match.group()) if match else None
        except (ValueError, TypeError):
            parsed = None
        ids = [o["option_id"] for o in task["options"]]
        if not isinstance(parsed, dict) or parsed.get("selected_option_id") not in ids:
            return {**accounting, "status": "error", "selected_option_id": None, "probabilities": None,
                    "causes_cited": [], "error": "Invalid verbalized-confidence JSON/option"}
        confidence = parsed.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            return {**accounting, "status": "error", "selected_option_id": None, "probabilities": None,
                    "causes_cited": [], "error": "Invalid verbalized confidence"}
        selected = parsed["selected_option_id"]
        scores = {key: confidence if key == selected else (1 - confidence) / (len(ids) - 1) for key in ids}
        cited = parsed.get("causes_cited", [])
        if not isinstance(cited, list) or not all(isinstance(x, str) for x in cited):
            cited = []
        return {**accounting, "selected_option_id": selected, "probabilities": scores, "status": "ok", "causes_cited": cited}

    def predict_sampling(self, task, *, samples=10, seed=0, temperature=0.8):
        """Repeated constrained one-token answer sampling, not CoT ensembles.

        Sharing the deterministic prompt prefill is exactly equivalent to
        repeating it. Raw draws and RNG seed make the Monte Carlo estimate auditable.
        """
        if samples < 10:
            raise ValueError("The sampling track requires at least ten draws")
        start = time.perf_counter()
        ids, scores, tokens = self.label_scores(task)
        distribution = self.torch.tensor(scores, dtype=self.torch.float64)
        distribution = (distribution.clamp_min(1e-30).log() / temperature).softmax(-1)
        generator = self.torch.Generator(device="cpu").manual_seed(seed)
        draws = self.torch.multinomial(distribution, samples, replacement=True, generator=generator).tolist()
        counts = Counter(ids[index] for index in draws)
        probabilities = {key: counts[key] / samples for key in ids}
        selected = max(ids, key=lambda key: probabilities[key])
        return {"selected_option_id": selected, "probabilities": probabilities, "status": "ok", "causes_cited": [],
                "input_tokens": tokens, "output_tokens": samples, "api_cost_usd": 0.0,
                "gpu_seconds": time.perf_counter() - start if self.device.startswith("cuda") else None,
                "metadata": {**self.identity, "algorithm": "constrained one-token sampling", "n": samples, "seed": seed,
                             "temperature": temperature, "draws": [ids[index] for index in draws],
                             "prefill_shared": True, "reasoning_ensemble": False}}

    def predict(self, task):
        return self.predict_logprobs(task)

    def next_action(self, public_view, public_history):
        # Tool track receives no menus (to test actual action construction).
        started = time.perf_counter()
        self.last_action_usage = {"input_tokens": 0, "output_tokens": 0, "api_cost_usd": 0.0,
                                  "gpu_seconds": 0.0, "input_tokens_known": True, "output_tokens_known": True}
        task = dict(public_view)
        task["options"] = []
        task["history"] = list(task.get("history", [])) + public_history
        inputs = self.inputs(task, 'Return ONLY one JSON action with kind invoke/read/wait/escalate/finish. For invoke supply capability_id and arguments; for read optionally state_ids; for wait dt_s. Finish only after observing the goal. Public capabilities only.')
        self.last_action_usage["input_tokens"] = inputs.input_ids.shape[-1]
        try:
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=192, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
            suffix = generated[0, inputs.input_ids.shape[-1]:]
            self.last_action_usage["output_tokens"] = len(suffix)
        except Exception:
            self.last_action_usage["output_tokens_known"] = False
            raise
        finally:
            if self.device.startswith("cuda"):
                self.torch.cuda.synchronize(self.device)
                self.last_action_usage["gpu_seconds"] = time.perf_counter()-started
        raw = self.tokenizer.decode(suffix, skip_special_tokens=True)
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("Tool completion lacks a JSON action")
        return json.loads(match.group())


class FrozenHiddenStateEncoder:
    """Chunk-pooled final hidden states from an untrained/frozen local LLM."""
    def __init__(self, llm, chunk_tokens=1024):
        self.llm, self.chunk_tokens = llm, chunk_tokens
        self.identity = {**llm.identity, "encoder": "final-layer mean pooled hidden states", "chunk_tokens": chunk_tokens,
                         "chunk_pooling": "token-weighted mean across ALL chunks", "fitted_on_labels": False}
        self.cache = {}
        self.last_usage = {}

    def encode(self, texts):
        import numpy as np
        rows = []
        computed_tokens = cache_hits = 0
        for text in texts:
            if text not in self.cache:
                tokens = self.llm.tokenizer.encode(text, add_special_tokens=False) or [self.llm.tokenizer.eos_token_id]
                computed_tokens += len(tokens)
                total = None
                for start in range(0, len(tokens), self.chunk_tokens):
                    chunk = self.llm.torch.tensor([tokens[start:start + self.chunk_tokens]], device=self.llm.device)
                    with self.llm.torch.inference_mode():
                        # Base model avoids materializing an unnecessary vocabulary head.
                        output = self.llm.model.model(input_ids=chunk, use_cache=False)
                        vector = output.last_hidden_state[0].float().sum(0).cpu().numpy()
                    total = vector if total is None else total + vector
                vector = total / len(tokens)
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
                self.cache[text] = vector
            else:
                cache_hits += 1
            rows.append(self.cache[text])
        self.last_usage = {"content_tokens": computed_tokens, "cache_hits": cache_hits,
                           "texts": len(rows), "truncated_tokens": 0,
                           "cache_policy": "exact public text; fit and predict share frozen content features"}
        return np.stack(rows)
