"""Reversible public JSON factoring for local graph model inputs.

This module does not choose evidence, actions, fields or time windows. The caller
must provide an already-public value. Contracts, schemas, native responses,
corrections, timestamps and every offered preset survive reconstruction exactly
as JSON values. Object member order is immaterial; array order and numeric types
are preserved. Nothing is loaded from an evaluator or an external archive.

``data`` remains readable JSON. ``$public_ref`` resolves into ``shared`` and
``$public_table`` is a columnar list of objects. Table paths identify nested
object fields; each row is [presence_bitmask, present_values]. Defaults apply to
every row. Missing fields therefore stay distinct from explicit nulls. Literal
objects containing the reserved marker names are escaped, not misinterpreted.

Factoring is not a promise that a complete manual fits a particular model. A
decision-specific public projection must be declared separately by its caller.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math


PROTOCOL = "lossless-shared-public-json-v1"
REFERENCE = "$public_ref"
TABLE = "$public_table"
OBJECT = "$public_object"
PUBLIC_CONTEXT_INSTRUCTION = (
    "The public input uses lossless-shared-public-json-v1. Read data as the payload. "
    "An object {$public_ref:name} means the value named in shared, recursively. "
    "An object {$public_table:{paths,defaults,rows}} means an ordered array of objects: "
    "each path is a list of nested object keys, defaults are [path,value] pairs applied "
    "to every row, and a row [mask,values] supplies paths[i] when bit i of mask is set. "
    "Values follow increasing path index. A clear bit means the field is absent, "
    "whereas an explicit null remains null. {$public_object:[[key,value],...]} is a "
    "literal object with those entries. All referenced values are supplied here; "
    "no evidence, timestamp, correction, schema or offered preset has been dropped. "
    "source_sha256 is an integrity identifier, not an observation or action ID."
)
_MARKERS = frozenset((REFERENCE, TABLE, OBJECT))
_FIELDS = frozenset(("protocol", "data", "shared", "source_sha256"))


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _validate_json(value, ancestors=None):
    ancestors = set() if ancestors is None else ancestors
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Public JSON contains a non-finite number")
        return
    if type(value) not in (dict, list):
        raise ValueError("Public context must contain JSON values only")
    identity = id(value)
    if identity in ancestors:
        raise ValueError("Public JSON contains a cycle")
    ancestors.add(identity)
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("Public JSON object keys must be strings")
            children = value.values()
        else:
            children = value
        for child in children:
            _validate_json(child, ancestors)
    finally:
        ancestors.remove(identity)


def _fingerprint(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _walk(value):
    yield value
    if type(value) is dict:
        for child in value.values():
            yield from _walk(child)
    elif type(value) is list:
        for child in value:
            yield from _walk(child)


def _leaves(record, prefix=()):
    """Flatten objects only; empty objects and all arrays remain leaf values."""
    for key, value in record.items():
        path = (*prefix, key)
        if type(value) is dict and value:
            yield from _leaves(value, path)
        else:
            yield path, value


class _Encoder:
    def __init__(self, value):
        self.counts = Counter()
        for node in _walk(value):
            if type(node) in (dict, list, str):
                signature = _json(node)
                if len(signature) >= 80:
                    self.counts[signature] += 1
        self.names = {}
        self.shared = {}

    def encode(self, value, *, root=False, hint="shared"):
        if not root and type(value) in (dict, list, str):
            signature = _json(value)
            if self.counts[signature] > 1:
                if signature not in self.names:
                    label = hint if hint in {"contract", "observation", "response", "arguments", "manual"} else "shared"
                    name = f"{label}_{len(self.names) + 1}"
                    self.names[signature] = name
                    self.shared[name] = self.encode(value, root=True, hint=hint)
                return {REFERENCE: self.names[signature]}
        if type(value) is dict:
            pairs = [(key, self.encode(child, hint=key)) for key, child in value.items()]
            if _MARKERS.intersection(value):
                return {OBJECT: [[key, child] for key, child in pairs]}
            return dict(pairs)
        if type(value) is list:
            plain = [self.encode(child) for child in value]
            if len(value) >= 3 and all(type(child) is dict for child in value):
                table = self.table(value)
                if len(_json(table)) < len(_json(plain)):
                    return table
            return plain
        return value

    def table(self, records):
        flattened = [dict(_leaves(record)) for record in records]
        paths = list(dict.fromkeys(path for row in flattened for path in row))
        defaults = []
        columns = []
        for path in paths:
            if all(path in row for row in flattened) and len({_json(row[path]) for row in flattened}) == 1:
                defaults.append([list(path), self.encode(flattened[0][path], hint=path[-1])])
            else:
                columns.append(path)
        rows = []
        for record in flattened:
            present = [(index, path) for index, path in enumerate(columns) if path in record]
            rows.append([sum(1 << index for index, _ in present),
                         [self.encode(record[path], hint=path[-1]) for _, path in present]])
        return {TABLE: {"paths": [list(path) for path in columns], "defaults": defaults, "rows": rows}}


def _references(value):
    if type(value) is dict:
        if set(value) == {REFERENCE}:
            yield value[REFERENCE]
        else:
            for child in value.values():
                yield from _references(child)
    elif type(value) is list:
        for child in value:
            yield from _references(child)


def _replace_reference(value, name, replacement):
    if type(value) is dict:
        if value == {REFERENCE: name}:
            return deepcopy(replacement)
        return {key: _replace_reference(child, name, replacement) for key, child in value.items()}
    if type(value) is list:
        return [_replace_reference(child, name, replacement) for child in value]
    return value


def _prune(data, shared):
    """Discard trial-table leftovers and inline references that add size."""
    while True:
        reachable = set()
        pending = list(_references(data))
        while pending:
            name = pending.pop()
            if name not in reachable:
                reachable.add(name)
                pending.extend(_references(shared[name]))
        shared = {name: value for name, value in shared.items() if name in reachable}
        counts = Counter(_references(data))
        for value in shared.values():
            counts.update(_references(value))
        removable = None
        for name, value in shared.items():
            reference_length = len(_json({REFERENCE: name}))
            entry_length = len(_json({name: value})) - 2
            if counts[name] * len(_json(value)) <= counts[name] * reference_length + entry_length:
                removable = name
                break
        if removable is None:
            return data, shared
        value = shared.pop(removable)
        data = _replace_reference(data, removable, value)
        shared = {name: _replace_reference(child, removable, value) for name, child in shared.items()}


def compact_json(value):
    """Factor a supplied public JSON value, checking exact reconstruction."""
    _validate_json(value)
    encoder = _Encoder(value)
    data, shared = _prune(encoder.encode(value, root=True), encoder.shared)
    result = {"protocol": PROTOCOL, "data": data, "shared": shared,
              "source_sha256": _fingerprint(value)}
    restored = restore_json(result)
    if _json(restored) != _json(value):
        raise ValueError("Public context factoring changed its source JSON")
    return result


def _path(value):
    if type(value) is not list or not value or any(type(part) is not str for part in value):
        raise ValueError("Invalid public table object path")
    return tuple(value)


def _assign(record, path, value, assigned):
    if any(path[:len(old)] == old or old[:len(path)] == path for old in assigned):
        raise ValueError("Overlapping public table object paths")
    assigned.add(path)
    target = record
    for part in path[:-1]:
        target = target.setdefault(part, {})
        if type(target) is not dict:
            raise ValueError("Conflicting public table object paths")
    target[path[-1]] = value


def restore_json(envelope):
    """Reconstruct JSON; reject broken references, cycles and altered contents."""
    _validate_json(envelope)
    if type(envelope) is not dict or set(envelope) != _FIELDS or envelope["protocol"] != PROTOCOL:
        raise ValueError("Unsupported compact public context envelope")
    shared = envelope["shared"]
    if type(shared) is not dict or any(type(name) is not str for name in shared):
        raise ValueError("Invalid shared public value catalog")
    active = set()
    cache = {}

    def decode(value):
        if type(value) is list:
            return [decode(child) for child in value]
        if type(value) is not dict:
            return value
        markers = _MARKERS.intersection(value)
        if not markers:
            return {key: decode(child) for key, child in value.items()}
        if len(value) != 1 or len(markers) != 1:
            raise ValueError("Ambiguous public JSON encoding marker")
        if REFERENCE in value:
            name = value[REFERENCE]
            if type(name) is not str or name not in shared:
                raise ValueError("Missing shared public value")
            if name in active:
                raise ValueError("Cyclic shared public value")
            if name not in cache:
                active.add(name)
                try:
                    cache[name] = decode(shared[name])
                finally:
                    active.remove(name)
            return deepcopy(cache[name])
        if OBJECT in value:
            pairs = value[OBJECT]
            if type(pairs) is not list:
                raise ValueError("Invalid escaped public object")
            record = {}
            for pair in pairs:
                if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str or pair[0] in record:
                    raise ValueError("Invalid escaped public object entry")
                record[pair[0]] = decode(pair[1])
            return record
        table = value[TABLE]
        if type(table) is not dict or set(table) != {"paths", "defaults", "rows"}:
            raise ValueError("Invalid public object table")
        if any(type(table[key]) is not list for key in table):
            raise ValueError("Invalid public object table arrays")
        paths = [_path(path) for path in table["paths"]]
        defaults = []
        for entry in table["defaults"]:
            if type(entry) is not list or len(entry) != 2:
                raise ValueError("Invalid public object table default")
            defaults.append((_path(entry[0]), entry[1]))
        all_paths = paths + [path for path, _ in defaults]
        if len(set(all_paths)) != len(all_paths):
            raise ValueError("Repeated public object table path")
        restored = []
        for row in table["rows"]:
            if type(row) is not list or len(row) != 2:
                raise ValueError("Invalid public object table row")
            mask, cells = row
            if type(mask) is not int or mask < 0 or mask.bit_length() > len(paths) or type(cells) is not list:
                raise ValueError("Invalid public object table presence mask")
            if mask.bit_count() != len(cells):
                raise ValueError("Public object table cell count disagrees with mask")
            record, assigned = {}, set()
            for path, child in defaults:
                _assign(record, path, decode(child), assigned)
            values = iter(cells)
            for index, path in enumerate(paths):
                if mask & (1 << index):
                    _assign(record, path, decode(next(values)), assigned)
            restored.append(record)
        return restored

    result = decode(envelope["data"])
    if type(envelope["source_sha256"]) is not str or _fingerprint(result) != envelope["source_sha256"]:
        raise ValueError("Reconstructed public context fingerprint mismatch")
    return result


def compact_public_context(context):
    """Factor a public benchmark context without changing its native inputs."""
    if type(context) is not dict:
        raise ValueError("Public benchmark context must be an object")
    return compact_json(context)


def restore_public_context(context):
    result = restore_json(context)
    if type(result) is not dict:
        raise ValueError("Restored public benchmark context must be an object")
    return result


def assert_public_context_roundtrip(original, compacted):
    """Validate a caller's original/public compact pair, including numeric types."""
    _validate_json(original)
    if _json(restore_public_context(compacted)) != _json(original):
        raise ValueError("Compact public context differs from the supplied original")
