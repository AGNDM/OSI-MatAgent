"""Reset and replay recorded requests, checking native/custom outputs exactly."""
import argparse
import json
from pathlib import Path

from app.lab_adapter import LabAdapter
from app.models import ActionProposal
from app.scenarios import case_by_id


def replay(case_id: str, trace_path: Path, split: str = "dev") -> int:
    lab = LabAdapter(case_by_id(case_id, split))
    count = 0
    for line in trace_path.read_text().splitlines():
        event = json.loads(line)
        if event["component"] != "execute" or event["kind"] != "tool_result":
            continue
        actual = lab.execute(ActionProposal.model_validate(event["proposal"]))
        for key in ["step", "proposal", "result", "revision", "source"]:
            if actual[key] != event[key]:
                raise ValueError(f"Replay diverged at action {count + 1}: {key}")
        count += 1
    if not count:
        raise ValueError("Trace contains no executed actions")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()
    print(f"Replay matched {replay(args.case_id, args.trace, args.split)} actions")


if __name__ == "__main__":
    main()
