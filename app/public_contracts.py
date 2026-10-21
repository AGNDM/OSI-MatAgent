"""Lossless sharing of repeated native metadata; parameter schemas stay inline."""
from collections import Counter
from copy import deepcopy
import json


def compact_capabilities(capabilities):
    counts=Counter()
    def identity(value):
        if isinstance(value,dict) and value and not any(isinstance(v,(dict,list)) for v in value.values()):
            encoded=json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))
            if len(encoded)>=70:return encoded
        return None
    def collect(value):
        key=identity(value)
        if key:counts[key]+=1
        if isinstance(value,dict):
            for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    for cap in capabilities.values():
        for field,value in cap.items():
            if field!='input_schema':collect(value)
    keys=sorted(k for k,n in counts.items() if n>1)
    names={k:f'm{i}' for i,k in enumerate(keys)}
    definitions={names[k]:json.loads(k) for k in keys}
    def encode(value):
        key=identity(value)
        if key in names:return {'$public_meta':names[key]}
        if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
        if isinstance(value,list):return [encode(v) for v in value]
        return deepcopy(value)
    result={name:{k:deepcopy(v) if k=='input_schema' else encode(v) for k,v in cap.items()}
            for name,cap in capabilities.items()}
    return result,definitions


def expand_capabilities(capabilities,definitions):
    def decode(value):
        if isinstance(value,dict) and set(value)=={'$public_meta'}:return deepcopy(definitions[value['$public_meta']])
        if isinstance(value,dict):return {k:decode(v) for k,v in value.items()}
        if isinstance(value,list):return [decode(v) for v in value]
        return deepcopy(value)
    return decode(capabilities)
