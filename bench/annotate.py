"""Evidence-linked machine suggestions, explicitly separate from human error review."""
import argparse
import csv
import json
from pathlib import Path

CODES = {"E1": "Observation", "E2": "Domain reasoning", "E3": "Planning", "E4": "Tool use",
    "E5": "Coordination", "E6": "Routing", "E7": "Verification", "E8": "Budget/termination", "E9": "Infrastructure/harness"}


def suggestions(record, events):
    findings = []
    for event in events:
        if event.get("error_code"):
            findings.append((event, event["error_code"], event.get("error", event["kind"])))
    if record["score"].get("false_success"):
        event = next((e for e in events if e["kind"] == "role_output" and e["component"] == "verification"
                      and e["output"]["status"] == "supported"), events[-1] if events else {})
        findings.append((event, "E7", "Success claim failed independent evidence/state assertions"))
    reason = record["report"]["terminal_reason"]
    if reason in {"llm_budget", "action_budget", "graph_budget", "deadline"} and not any(c == "E8" for _, c, _ in findings):
        findings.append((events[-1] if events else {}, "E8", reason))
    order = {e["event_id"]: i for i, e in enumerate(events)}
    findings.sort(key=lambda f: order.get(f[0].get("event_id"), len(events)))
    result = []
    for index, (event, code, observed) in enumerate(findings):
        result.append({"run_id": record["run_id"], "step_id": event.get("event_id", ""),
            "component": event.get("component", "harness"), "error_code": code,
            "evidence_ids": json.dumps(event.get("output", {}).get("evidence_ids", [])),
            "expected_behavior": "Honor the typed protocol, budget, and independent evidence criteria",
            "observed_behavior": observed,
            "primary_or_downstream": "first_observed_candidate" if index == 0 else "downstream_candidate",
            "reviewer_status": "automatic suggestion; semantic/causal attribution unreviewed",
            "system_recovered": record["score"]["task_success"]})
    if not result and not record["score"]["task_success"]:
        result.append({"run_id": record["run_id"], "step_id": events[-1]["event_id"] if events else "",
            "component": "uncertain", "error_code": "uncertain", "evidence_ids": "[]",
            "expected_behavior": "Family-specific independent task success",
            "observed_behavior": "Task failed without a mechanically attributable error",
            "primary_or_downstream": "first_observed_candidate", "reviewer_status": "requires semantic review",
            "system_recovered": False})
    return result


def analyze(directory):
    records = [json.loads(s) for s in (directory / "runs.jsonl").read_text().splitlines()]
    traces, annotations = {}, []
    for record in records:
        # Resolve only the artifact directory's trace, never an arbitrary path from a record.
        path = directory / "traces" / f"{record['run_id']}.jsonl"
        events = [json.loads(s) for s in path.read_text().splitlines()]
        traces[record["run_id"]] = events
        annotations.extend(suggestions(record, events))
    fields = ["run_id", "step_id", "component", "error_code", "evidence_ids", "expected_behavior", "observed_behavior", "primary_or_downstream", "reviewer_status", "system_recovered"]
    with (directory / "errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(annotations)
    configs = sorted({(r["role_backend"], r["decision_backend"]) for r in records})
    heatmap = []
    for role, router in configs:
        subset = [r for r in records if (r["role_backend"], r["decision_backend"]) == (role, router)]
        ids = {r["run_id"] for r in subset}
        heatmap.append({"roles": role, "router": router, "attempted": len(subset),
            "success": sum(r["score"]["task_success"] for r in subset),
            "counts": {code: len({a["run_id"] for a in annotations if a["run_id"] in ids and a["error_code"] == code}) for code in CODES}})
    contrast = next(((a, b) for a in records for b in records if a["case_id"] == b["case_id"]
        and a["repeat"] == b["repeat"] and a["score"]["task_success"] != b["score"]["task_success"]), None)
    selected = list(contrast) if contrast else records[:2]
    lines = ["# Trace inspection", "", "Automatic evidence-linked suggestions, not reviewed causal ground truth. No independent human reviewers have annotated these traces.", "",
        "Matched outcome contrast found." if contrast else "No matched success/failure contrast exists in these results; selected traces must not be presented as such."]
    for r in selected:
        lines += ["", f"## {r['case_id']} / {r['role_backend']} / {r['decision_backend']}", "",
                  f"Run `{r['run_id']}`; success={r['score']['task_success']}; terminal={r['report']['terminal_reason']}."]
        lines += [f"- `{a['step_id']}` {a['error_code']}: {a['observed_behavior']} ({a['reviewer_status']})"
                  for a in annotations if a["run_id"] == r["run_id"]]
        if not any(a["run_id"] == r["run_id"] for a in annotations):
            lines.append("No mechanical errors detected; this does not establish absence of semantic errors.")
    (directory / "trace-review.md").write_text("\n".join(lines) + "\n")
    data = json.dumps({"records": records, "traces": traces, "heatmap": heatmap, "annotations": annotations})
    # JSON is data, never executable HTML; public traces only, no evaluator snapshots.
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = '''<!doctype html><meta charset="utf-8"><title>Laboratory experiment traces</title>
<style>body{font:16px system-ui;margin:2rem;max-width:1200px;background:#f5f7fa;color:#132b40}#heatmap{overflow-x:auto}table{border-collapse:collapse;background:white;width:100%;table-layout:fixed;min-width:600px}td,th{border:1px solid #ccd6df;padding:.5rem;overflow-wrap:anywhere}td:first-child,th:first-child{width:30%}select{max-width:100%;padding:.5rem}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{background:white;margin:.5rem 0;padding:.75rem}h2{margin-top:2rem}</style>
<h1>Laboratory experiment results</h1><p>Recorded runs / replay inspection. No live model calls from this page.</p>
<p>Errors are automatic suggestions pending human semantic review. Heatmap cells count episodes with each label / all attempted episodes; labels may overlap.</p>
<div id="heatmap"></div><h2>Public trace timeline</h2><select id="runs"></select><pre id="outcome"></pre><div id="timeline"></div>
<script type="application/json" id="data">DATA</script><script>
const d=JSON.parse(document.getElementById('data').textContent), el=id=>document.getElementById(id), codes=Array.from({length:9},(_,i)=>'E'+(i+1));
const table=document.createElement('table'), head=table.insertRow();
['Role model / router','Success / attempts',...codes].forEach(t=>{let c=document.createElement('th');c.textContent=t;head.append(c)});
d.heatmap.forEach(r=>{let row=table.insertRow();[r.roles+' / '+r.router,r.success+'/'+r.attempted,...codes.map(c=>r.counts[c]+'/'+r.attempted)].forEach((v,i)=>{let c=row.insertCell();c.textContent=v;if(i>1)c.style.backgroundColor='rgba(215,80,60,'+(r.counts[codes[i-2]]/r.attempted*.6)+')'})});el('heatmap').append(table);
d.records.forEach(r=>{let o=document.createElement('option');o.value=r.run_id;o.textContent=r.case_id+' / '+r.role_backend+' / '+r.decision_backend+' / repeat '+r.repeat;el('runs').append(o)});
function show(){let id=el('runs').value,r=d.records.find(r=>r.run_id===id);el('outcome').textContent=JSON.stringify({score:r.score,report:r.report,elapsed_ms:r.elapsed_ms,fallback_count:r.fallback_count},null,2);el('timeline').replaceChildren();d.traces[id].forEach(e=>{let box=document.createElement('details'),title=document.createElement('summary'),pre=document.createElement('pre');title.textContent=e.event_id.split(':').at(-1)+' · '+e.component+' · '+e.kind;pre.textContent=JSON.stringify(e,null,2);box.append(title,pre);el('timeline').append(box)})}el('runs').onchange=show;if(d.records.length)show();
</script>'''.replace("DATA", data)
    (directory / "report.html").write_text(page)
    (directory / "error-counts.json").write_text(json.dumps(heatmap, indent=2) + "\n")
    return {"runs": len(records), "suggested_labels": len(annotations), "matched_contrast": contrast is not None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    print(json.dumps(analyze(args.directory)))


if __name__ == "__main__":
    main()
