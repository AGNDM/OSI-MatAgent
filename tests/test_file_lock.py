"""Exercise ledger serialization in real spawned processes on either OS."""
import multiprocessing

import pytest

from app.token_budget import TokenLedger


def _write_reservations(path, count):
    ledger = TokenLedger(path)
    for _ in range(count):
        reservation = ledger._reserve_bound("lock-test", 1)
        ledger.settle(reservation, {
            "prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1,
        })


def _join(processes):
    try:
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)


def test_ledger_serializes_spawned_processes(tmp_path):
    path = tmp_path / "ledger.jsonl"
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_write_reservations, args=(path, 20))
                 for _ in range(4)]
    for process in processes:
        process.start()
    _join(processes)
    snapshot = TokenLedger(path).snapshot()
    assert snapshot["requests_reserved"] == snapshot["requests_settled"] == 80
    assert snapshot["charged_tokens"] == 80
    assert snapshot["inflight_or_crashed"] == 0


def test_ledger_unlocks_after_exception(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = TokenLedger(path)
    with pytest.raises(RuntimeError):
        with ledger._locked():
            raise RuntimeError("release the lock on exit")
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_write_reservations, args=(path, 1))
    process.start()
    _join([process])
    assert ledger.snapshot()["requests_settled"] == 1
