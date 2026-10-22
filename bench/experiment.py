"""Shared request accounting and provider-stop state for hosted MCQA."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from app.token_budget import TokenLedger
from bench.instrument_eval_langgraph import SharedRequestPacer

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

def stop_reason(call):
    status = call.get('http_status')
    response = str(call.get('response', '')).lower()
    if status in (401, 402, 403, 404):
        return 'provider_unavailable_' + str(status)
    if status == 429 and any(word in response for word in ('daily', 'quota', 'limit_free', 'credits', '1000')):
        return 'provider_daily_limit'
    return None

class Experiment:
    def __init__(self, output, config, rate_state, external_stop=None):
        self.output, self.config = output, config
        self.ledger = TokenLedger(output / 'api-ledger.jsonl')
        self.pacer = SharedRequestPacer(rate_state, 3.4)
        self.stop = output / 'provider-stop.json'
        self.external_stop = external_stop

    def stopped(self):
        return self.stop.exists() or bool(self.external_stop and self.external_stop.exists())

    def record_stop(self, call):
        reason = stop_reason(call)
        if reason and not self.stop.exists():
            save(self.stop, {'reason': reason, 'at': now(), 'http_status': call.get('http_status'),
                         'response': call.get('response'), 'paid_fallback': False})
