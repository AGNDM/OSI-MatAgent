"""Read Slurm allocation evidence and estimate retained MCQA worker costs.

This module performs no inference. Rates describe allocated GPU units (a MIG
slice is one unit), and estimates are not invoices or active GPU kernel costs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess


ORIGINAL_TARIFF_PATH = Path("/etc/hpc_motd/02-config_opt.yaml")
RENDERER_PATH = Path("/usr/share/scitas_pyapp/python/lib/python3.13/site-packages/hpc_motd/pricing.py")
FIELDS = "JobIDRaw,JobID,Partition,NodeList,State,ElapsedRaw,AllocTRES,AllocCPUS,ReqMem,TotalCPU"
TERMINAL = frozenset({
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT",
})
SCOPE = (
    "Retained authenticated MCQA shard allocations for the full method cohort "
    "(normally 388 questions; T1 reference 208), including startup, model loading "
    "and allocation overhead. Excludes sequential/graph runs, environment setup, "
    "model downloads, feature fitting, domain folds and discarded attempts. "
    "Allocation time cannot be attributed exactly to held-out or category/menu-size subgroups."
)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _unquote(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _tariffs(tariff_path):
    """Parse only the four documented price blocks, without loading other configs."""
    metadata = {
        "source_path": str(tariff_path), "original_site_source_path": str(ORIGINAL_TARIFF_PATH),
        "source_capture": "Unmodified local SCITAS tariff configuration copied before the export job.",
        "currency": "CHF",
        "rate_unit": "CHF per allocated GPU unit-hour",
        "estimate_not_invoice": True, "rates_by_partition": {},
        "note": "MIG units are allocated slices, not full-GPU equivalents. Recorded charges may differ, including failure-related billing policies.",
    }
    try:
        raw = Path(tariff_path).read_bytes()
        metadata["source_sha256"] = _hash(raw)
        text = raw.decode("utf-8")
        header = re.search(r"(?m)^  header:\s*(.*?)\s*$", text)
        metadata["header"] = _unquote(header.group(1)) if header else None
        blocks = list(re.finditer(r"(?m)^    (h100|l40s|mig12gb|mig24gb):[ \t]*$", text))
        rates = {}
        for index, match in enumerate(blocks):
            name = match.group(1)
            if name in rates:
                raise ValueError("Repeated tariff partition: " + name)
            end = blocks[index + 1].start() if index + 1 < len(blocks) else len(text)
            values = {}
            for key, value in re.findall(r"(?m)^      (value|unit|multiplier|prefix):[ \t]*(.*?)[ \t]*$", text[match.end():end]):
                if key in values:
                    raise ValueError("Repeated tariff field: " + name + "/" + key)
                values[key] = _unquote(value)
            if set(values) != {"value", "unit", "multiplier", "prefix"}:
                raise ValueError("Incomplete tariff block: " + name)
            if values["prefix"] != "CHF" or values["unit"] != "/gpu/hour":
                raise ValueError("Unsupported tariff currency/unit: " + name)
            value, multiplier = Decimal(values["value"]), Decimal(values["multiplier"])
            price = value * multiplier
            if not all(number.is_finite() and number > 0 for number in (value, multiplier, price)):
                raise ValueError("Invalid tariff value: " + name)
            if not math.isfinite(float(price)):
                raise ValueError("Tariff cannot be represented as finite JSON: " + name)
            rates[name] = float(price)
        if set(rates) != {"h100", "l40s", "mig12gb", "mig24gb"}:
            raise ValueError("Expected all four GPU tariff partitions")
        metadata["rates_by_partition"] = rates
        metadata["status"] = "read_from_installed_site_configuration"
        try:
            metadata["site_price_renderer"] = {"path": str(RENDERER_PATH), "sha256": _hash(RENDERER_PATH.read_bytes())}
        except OSError:
            pass
        return metadata, None
    except (OSError, UnicodeError, ValueError, InvalidOperation) as error:
        metadata["status"] = "unavailable"
        metadata["error"] = str(error)
        return metadata, "Tariffs unavailable: " + str(error)


def _aggregate(records):
    expected = len(records)
    matched = [row for row in records if row.get("matched")]
    hours = [row["gpu_allocation_hours"] for row in records if row.get("gpu_allocation_hours") is not None]
    costs = [row["gpu_cost_chf_estimate"] for row in records if row.get("gpu_cost_chf_estimate") is not None]
    complete_hours = expected > 0 and len(hours) == expected
    complete_costs = expected > 0 and len(costs) == expected
    return {
        "job_ids": [row["job_id"] for row in records],
        "expected_job_count": expected, "matched_job_count": len(matched),
        "terminal_job_count": sum(row.get("terminal") is True for row in records),
        "matched_job_coverage": len(matched) / expected if expected else None,
        "allocation_job_coverage": len(hours) / expected if expected else None,
        "cost_estimate_job_coverage": len(costs) / expected if expected else None,
        "gpu_allocation_hours": sum(hours) if complete_hours else None,
        "gpu_cost_chf_estimate": sum(costs) if complete_costs else None,
        "known_partial_gpu_allocation_hours": sum(hours) if hours else None,
        "known_partial_gpu_cost_chf_estimate": sum(costs) if costs else None,
        "allocation_accounting_complete": complete_hours,
        "cost_estimate_complete": complete_costs,
        "states": dict(sorted(Counter(row.get("state", "missing") for row in records).items())),
        "unavailable_reasons": dict(sorted(Counter(
            row["unavailable_reason"] for row in records if row.get("unavailable_reason")
        ).items())),
        "currency": "CHF", "estimate_not_invoice": True, "scope": SCOPE,
    }


def _with_partitions(records):
    result = _aggregate(records)
    partitions = defaultdict(list)
    for row in records:
        partitions[row.get("partition") or "unknown"].append(row)
    result["by_partition"] = {key: _aggregate(value) for key, value in sorted(partitions.items())}
    result["partition_coverage_note"] = "Partition summaries cover the listed workers only; unmatched jobs remain in unknown."
    return result


def collect_accounting(source_jobs, workdir, *, tariff_path):
    """Return (global evidence, per-experiment totals); unavailable data stays null."""
    rates, tariff_error = _tariffs(tariff_path)
    errors = [tariff_error] if tariff_error else []
    records, expected = {}, {}
    for job_id, source in source_jobs.items():
        job_id = str(job_id)
        expected[job_id] = dict(source)
        records[job_id] = {
            "job_id": job_id, "experiment": source.get("experiment"), "track": source.get("track"),
            "source_path": str(source.get("source_path", "")),
            "source_compute_node": source.get("compute_node"), "matched": False,
            "terminal": False, "state": "missing", "gpu_allocation_hours": None,
            "gpu_cost_chf_estimate": None, "unavailable_reason": "No matching Slurm allocation row",
        }
    command = ["sacct", "-X", "--array", "-n", "-P", "-j", ",".join(sorted(expected, key=lambda value: int(value) if value.isdigit() else -1)), "-o", FIELDS]
    evidence = {"command": command, "timeout_seconds": 45}
    raw = stderr = ""
    can_query = bool(expected)
    if not all(re.fullmatch(r"[0-9]+", key) for key in expected):
        errors.append("Invalid raw job identifier; accounting query not executed")
        can_query = False
    if any(source.get("track") != "mcqa" for source in expected.values()):
        errors.append("Non-MCQA source supplied; accounting query not executed")
        can_query = False
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        errors.append("Accounting export requires a Slurm compute allocation")
        can_query = False
    if can_query:
        try:
            response = subprocess.run(command, text=True, capture_output=True, timeout=45, check=False)
            raw, stderr = response.stdout, response.stderr
            evidence["returncode"] = response.returncode
            if response.returncode:
                errors.append("sacct exited with status " + str(response.returncode))
            else:
                for line in raw.splitlines():
                    if not line.strip():
                        continue
                    fields = line.split("|")
                    if len(fields) < 10:
                        errors.append("Malformed sacct allocation row")
                        continue
                    jid, display, partition, node, state, elapsed, tres, cpus, memory, total_cpu = fields[:10]
                    if jid not in expected or "." in jid or "." in display:
                        continue
                    if records[jid]["matched"]:
                        errors.append("Repeated Slurm allocation row for raw job " + jid)
                        records[jid]["gpu_allocation_hours"] = None
                        records[jid]["gpu_cost_chf_estimate"] = None
                        records[jid]["unavailable_reason"] = "Ambiguous repeated allocation rows"
                        continue
                    row = records[jid]
                    normalized_state = state.split()[0].rstrip("+") if state.strip() else "unknown"
                    row.update(matched=True, display_job_id=display, partition=partition,
                               node_list=node, state=normalized_state, terminal=normalized_state in TERMINAL,
                               elapsed_seconds=int(elapsed) if elapsed.isdigit() else None,
                               alloc_tres=tres, allocated_cpus=int(cpus) if cpus.isdigit() else None,
                               requested_host_memory=memory, total_cpu_time=total_cpu)
                    gpu = re.findall(r"(?:^|,)gres/gpu=([0-9]+)(?=,|$)", tres)
                    row["allocated_gpu_units"] = int(gpu[0]) if len(gpu) == 1 else None
                    billing = re.findall(r"(?:^|,)billing=([0-9]+(?:\.[0-9]+)?)(?=,|$)", tres)
                    row["allocated_billing_units"] = float(billing[0]) if len(billing) == 1 else None
                    if not row["terminal"]:
                        row["unavailable_reason"] = "Allocation is not terminal; elapsed time is not final"
                    elif row["elapsed_seconds"] is None or row["allocated_gpu_units"] is None:
                        row["unavailable_reason"] = "ElapsedRaw or unambiguous generic gres/gpu allocation is unavailable"
                    else:
                        hours = row["elapsed_seconds"] * row["allocated_gpu_units"] / 3600
                        row["gpu_allocation_hours"] = hours
                        rate = rates.get("rates_by_partition", {}).get(partition)
                        row["rate_chf_per_gpu_unit_hour"] = rate
                        row["billing_unit_hours"] = (row["elapsed_seconds"] * row["allocated_billing_units"] / 3600
                                                     if row["allocated_billing_units"] is not None else None)
                        row["gpu_cost_chf_estimate"] = hours * rate if rate is not None else None
                        row["unavailable_reason"] = None if rate is not None else "No validated local CHF tariff for this partition"
        except subprocess.TimeoutExpired as error:
            raw = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
            errors.append("sacct timed out after 45 seconds; partial output is not used")
        except (OSError, ValueError) as error:
            errors.append("sacct accounting unavailable: " + str(error))
    try:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        for filename, value in (("sacct-raw.psv", raw), ("sacct.stderr", stderr)):
            path = workdir / filename
            path.write_text(value, encoding="utf-8")
            evidence[filename] = {"path": str(path), "sha256": _hash(value.encode("utf-8"))}
    except OSError as error:
        errors.append("Unable to preserve accounting stdout/stderr: " + str(error))
    rows = [records[key] for key in sorted(records, key=lambda value: int(value) if value.isdigit() else -1)]
    unavailable = Counter(row["unavailable_reason"] for row in rows if row.get("unavailable_reason"))
    if unavailable:
        errors.extend(f"{count} jobs: {reason}" for reason, count in sorted(unavailable.items()))
    global_accounting = _with_partitions(rows)
    global_accounting.update(
        collected_at_utc=datetime.now(timezone.utc).isoformat(),
        status="complete" if global_accounting["cost_estimate_complete"] else "partial_or_unavailable",
        raw_job_records=rows, tariff=rates, accounting_source=evidence,
        unavailable_errors=errors, hosted_api_cost_usd=0.0,
        definitions={
            "gpu_allocation_hours": "Sum of terminal allocation ElapsedRaw seconds times generic gres/gpu units, divided by 3600; includes MIG slice-hours, not physical full-GPU equivalents or active kernel time.",
            "gpu_cost_chf_estimate": "Terminal GPU allocation hours times the installed site partition tariff; an estimate, not a recorded charge or invoice.",
            "known_partial_totals": "Known terminal worker totals only; missing or nonterminal allocations are not treated as zero.",
            "scope": SCOPE,
        },
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["experiment"]].append(row)
    return global_accounting, {name: _with_partitions(value) for name, value in sorted(grouped.items(), key=lambda item: str(item[0]))}
