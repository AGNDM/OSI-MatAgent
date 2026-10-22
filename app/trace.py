from datetime import datetime, timezone
from copy import deepcopy
import json
from pathlib import Path


class Trace:
    def __init__(self, episode_id: str, path: Path | None = None, observer=None):
        self.episode_id = episode_id
        self.events: list[dict] = []
        self.path = path
        self.observer = observer
        self.observer_error_count = 0
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def emit(self, component: str, kind: str, **data):
        event = {"event_id": f"{self.episode_id}:{len(self.events) + 1}",
            "episode_id": self.episode_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": component, "kind": kind, **data}
        self.events.append(event)
        if self.path:
            with self.path.open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
        if self.observer:
            try:
                self.observer(deepcopy(event))
            except Exception:
                # Presentation failure cannot change the experiment or its stored evidence.
                self.observer_error_count += 1
