import json
import sys
import time

import httpx
import pytest

from app.demo import DemoController
from app.expert_config import CompatibleExpert, ExpertEndpoint, ExpertPlan
from app.models import VerificationReport
from app.scenarios import case_by_id
from bench.run import run_episode


def wait_finished(controller):
    deadline = time.monotonic() + 10
    while controller.state()["running"] and time.monotonic() < deadline:
        time.sleep(.01)
    assert not controller.state()["running"]


def test_demo_runs_isolated_world_reloads_and_replays_without_model_calls(tmp_path):
    controller = DemoController(tmp_path)
    with pytest.raises(ValueError):
        controller.start("dev-01", "openrouter")
    key = controller.start("dev-01", "scripted-rules")
    wait_finished(controller)
    state = controller.state()
    assert state["display_mode"] == "live"
    assert state["record"]["score"]["task_success"]
    assert any(e["kind"] == "role_start" for e in state["events"])
    assert all("llm_metadata" not in e and "input" not in e for e in state["events"])
    replayed = controller.replay(key)
    assert replayed["matched_actions"] == 3
    assert replayed["run_id"] == state["record"]["run_id"] and replayed["case_id"] == "dev-01"
    controller.reset()
    restored = DemoController(tmp_path)
    restored.view(key)
    assert restored.state()["display_mode"] == "recorded"
    assert restored.state()["record"]["run_id"] == state["record"]["run_id"]
    assert restored.replay(key)["model_calls"] == 0


def test_presentation_observer_failure_cannot_change_results_or_trace(tmp_path):
    def bad_observer(event):
        event.clear()
        raise RuntimeError("broken presentation")
    r = run_episode(case_by_id("dev-01"), tmp_path, observer=bad_observer)
    assert r["score"]["task_success"] and r["observer_error_count"] > 0
    events = [json.loads(s) for s in next((tmp_path / "traces").glob("*.jsonl")).read_text().splitlines()]
    assert all(e["event_id"] for e in events)


def test_general_expert_requires_explicit_distinct_local_models():
    with pytest.raises(ValueError, match="not configured"):
        CompatibleExpert(ExpertEndpoint(name="pending"))
    for url in ["https://openrouter.ai", "http://external.example"]:
        with pytest.raises(ValueError, match="loopback"):
            ExpertEndpoint(name="wrong", base_url=url, model="general")
    with pytest.raises(ValueError, match="general models"):
        ExpertEndpoint(name="wrong", model="some/model:free")
    with pytest.raises(ValueError, match="not independent"):
        ExpertPlan(reference=ExpertEndpoint(name="a", model="same", base_url="http://127.0.0.1:9001"),
            contrast=ExpertEndpoint(name="b", model="same", base_url="http://127.0.0.1:9001"))


def test_configured_expert_uses_declared_model_and_records_no_router_identity(monkeypatch):
    seen = []
    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "general-a", "choices": [{"message": {
            "content": '{"status":"unresolved","evidence_ids":[],"missing_evidence":["mass"]}'}}],
            "usage": {"completion_tokens": 10}})
    original = httpx.AsyncClient
    monkeypatch.setattr("app.expert_config.httpx.AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    backend = CompatibleExpert(ExpertEndpoint(name="ref", base_url="http://127.0.0.1:9001", model="general-a", revision="r1"))
    assert backend.invoke("verification", "protocol", {}, VerificationReport)["status"] == "unresolved"
    assert seen[0]["model"] == "general-a"
    assert backend.manifest()["is_openjev_router"] is False
    assert "OpenJev-Flash" not in json.dumps(backend.manifest())


def test_free_batch_is_refused_before_credentials_network_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr("app.openrouter.model_catalog", lambda: pytest.fail("must not access catalog"))
    from bench import run, matrix
    for module, args in [(run, ["--models", "a:free", "--split", "test"]),
                         (run, ["--models", "a:free", "b:free", "--cases", "dev-03"]),
                         (matrix, ["--models", "a:free", "b:free", "--execute"])]:
        output = tmp_path / module.__name__
        monkeypatch.setattr(sys, "argv", ["command", *args, "--output", str(output)])
        with pytest.raises(SystemExit):
            module.main()
        assert not output.exists()


def test_unconfigured_plan_has_48_entries_and_execute_fails_before_network(tmp_path, monkeypatch):
    from bench import matrix
    template = tmp_path / "experts.json"
    template.write_text(json.dumps({"reference": {"name": "ref"}, "contrast": {"name": "contrast"}}))
    output = tmp_path / "plan"
    argv = ["matrix", "--config", str(template), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    matrix.main()
    assert len(json.loads((output / "freeze.json").read_text())["plan"]) == 48
    assert len(json.loads((output / "status.json").read_text())["unconfigured_experts"]) == 2
    monkeypatch.setattr(sys, "argv", [*argv, "--execute", "--resume"])
    with pytest.raises(SystemExit):
        matrix.main()
    assert not (output / "runs.jsonl").exists()


def test_cause_coverage_is_not_named_final_diagnosis_accuracy():
    from bench.score import score
    from app.lab_adapter import LabAdapter
    fixture = case_by_id("dev-01")
    hypotheses = [{"cause": cause, "supporting_ids": ["obs-1"], "contradicting_ids": []}
                  for cause in ["process_mismatch", "pan_contamination", "none", "missing_evidence"]]
    result = score(fixture, LabAdapter(fixture).evaluator_snapshot(), {"hypotheses": hypotheses})
    assert result["expected_cause_covered"]
    assert "diagnosis_correct" not in result
    assert not result["task_success"]


def test_browser_rejects_cloud_expert_before_output_or_transport(tmp_path, monkeypatch):
    endpoint = ExpertEndpoint(name='cloud-test', provider='siliconflow',
        base_url='https://api.siliconflow.cn/v1', model='deepseek-ai/DeepSeek-V4.1-Flash',
        api_key_env='TEST_ONLY_KEY')
    monkeypatch.setattr('app.demo.CompatibleExpert', lambda *args: pytest.fail('Browser must not create cloud transport'))
    output = tmp_path/'cloud-disabled'
    with pytest.raises(ValueError, match='local experts only'):
        DemoController(output, expert=endpoint)
    assert not output.exists()


def test_browser_still_accepts_configured_loopback_expert_without_connecting(tmp_path):
    endpoint = ExpertEndpoint(name='local-test', base_url='http://127.0.0.1:9001', model='local-general')
    controller = DemoController(tmp_path, expert=endpoint)
    assert 'configured-openjev' in controller.modes
