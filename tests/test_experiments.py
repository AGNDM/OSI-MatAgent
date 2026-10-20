import json

from app.models import Limits
from app.retrieval import manifest, retrieve
from app.scenarios import case_by_id, load_cases
from bench.annotate import analyze, suggestions
from bench.matrix import schedule
from bench.routing_probe import snapshots
from bench.run import run_episode


def test_retrieval_is_fixed_public_protocol_and_never_fixture_truth(tmp_path):
    record = run_episode(case_by_id("dev-01"), tmp_path)
    events = [json.loads(s) for s in next((tmp_path / "traces").glob("*.jsonl")).read_text().splitlines()]
    for event in events:
        if event["kind"] == "role_output":
            passages = event["input"]["retrieval_evidence"]
            assert passages == retrieve(event["component"])
            text = json.dumps(passages)
            assert "initial_liquid_kg" not in text and "test-01" not in text
    assert record["score"]["task_success"]
    assert manifest()["limit"] == 2 and len(manifest()["source_hashes"]) > 1


def test_matrix_has_matched_reference_baseline_and_all_repeats():
    models = ["a:free", "b:free"]
    plan = schedule(models, load_cases("test"), 2)
    assert len(plan) == 48
    for case in load_cases("test"):
        for repeat in range(2):
            selected = [x for x in plan if x["case_id"] == case.case_id and x["repeat"] == repeat]
            assert {(x["model"], x["router"]) for x in selected} == {("a:free", "openjev"), ("b:free", "openjev"), ("a:free", "rules")}
    assert plan[0]["arm"] != plan[3]["arm"]


def test_probe_annotations_are_not_part_of_inputs_and_split_is_balanced():
    for split in ["dev", "test"]:
        items = snapshots(split)
        assert len(items) == 12
        assert all(sum(i["primary_route"] == r for i in items) == 3 for r in ["diagnose", "prepare_action", "verify", "deliberate"])
        assert all("acceptable_routes" not in i["summary"] for i in items)
    assert snapshots("dev")[0]["summary"] != snapshots("test")[0]["summary"]


def test_failed_attempts_and_budget_errors_remain_in_analysis(tmp_path):
    record = run_episode(case_by_id("dev-01"), tmp_path, limits=Limits(lab_actions=0))
    result = analyze(tmp_path)
    assert result["runs"] == 1 and not result["matched_contrast"]
    errors = (tmp_path / "errors.csv").read_text()
    assert "E8" in errors and "unreviewed" in errors
    counts = json.loads((tmp_path / "error-counts.json").read_text())[0]
    assert counts["attempted"] == 1 and counts["success"] == 0
    assert "No matched" in (tmp_path / "trace-review.md").read_text()


def test_unknown_semantic_failure_is_not_falsely_attributed():
    record = {"run_id": "x", "score": {"task_success": False}, "report": {"terminal_reason": "agent_unresolved"}}
    assert suggestions(record, [])[0]["error_code"] == "uncertain"


def test_matrix_freeze_resume_preserves_failed_attempts_and_refuses_drift(tmp_path, monkeypatch):
    import sys
    import pytest
    import bench.matrix as matrix
    from app.decision_backend import RulesBackend
    from app.roles import ScriptedRoles
    output = tmp_path / "matrix"
    config = tmp_path / "experts.json"
    config.write_text(json.dumps({"reference": {"name": "a", "model": "general-a", "base_url": "http://127.0.0.1:9001"},
        "contrast": {"name": "b", "model": "general-b", "base_url": "http://127.0.0.1:9002"}}))
    argv = ["matrix", "--config", str(config), "--repeats", "1", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv + ["--freeze-only"])
    matrix.main()
    assert not (output / "runs.jsonl").exists()

    class FakeRoles(ScriptedRoles):
        def __init__(self, config):
            self.backend_id = config.name
    monkeypatch.setattr(matrix, "CompatibleExpert", FakeRoles)
    monkeypatch.setattr(matrix, "LocalOpenJevBackend", RulesBackend)
    failing = [True]
    def episode(case, directory, limits, roles, decision, repeat):
        r = run_episode(case, limits=Limits(lab_actions=0), roles=roles, repeat=repeat)
        r["provider_failure_count"] = int(failing[0])
        with (directory / "runs.jsonl").open("a") as stream:
            stream.write(json.dumps(r) + "\n")
        return r
    monkeypatch.setattr(matrix, "run_episode", episode)
    monkeypatch.setattr(sys, "argv", argv + ["--resume", "--execute"])
    with pytest.raises(RuntimeError, match="Three consecutive"):
        matrix.main()
    assert json.loads((output / "status.json").read_text())["attempted"] == 3
    initial = (output / "runs.jsonl").read_text()
    failing[0] = False
    matrix.main()
    assert (output / "runs.jsonl").read_text().startswith(initial)
    assert json.loads((output / "status.json").read_text())["attempted"] == 24
    monkeypatch.setattr(matrix, "fingerprint", lambda: {"changed": True})
    with pytest.raises(SystemExit):
        matrix.main()
