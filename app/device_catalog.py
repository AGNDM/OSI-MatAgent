"""Public discovery cache; selection never loads reference tasks or native hosts."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

from app.device_registry import load_registry,open_native,public_capabilities

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'cases/devices/public-catalog.json'
SOURCES=['cases/devices/registry.json','app/device_registry.py','app/device_catalog.py','app/single_instrument.py','app/native_lifecycle.py',
         'vendor/chemlab/manifest.json','vendor/chemlab/expansion-code-manifest.json',
         'vendor/chemlab/expansion-data-manifest.json']


def sources():return {p:sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def build_cache(path=CACHE):
    """Developer-only offline discovery, never an agent-callable reset/setup tool."""
    from app.single_instrument import NativeLifecycle
    records=[]
    for spec in load_registry()['devices']:
        # Native constructors and declared environment fixtures are setup only.
        host,session,instance,description=open_native(spec)
        fields={k:deepcopy(v) for k,v in description.get('state_slots',{}).items()
                if v.get('visibility','agent')=='agent' and not k.startswith(('truth.','debug.','host.','plant.'))}
        lifecycle=NativeLifecycle(spec,host,session,instance,description)
        records.append({'id':spec['slug'],'asset_id':spec['asset_id'],'chem_id':spec['chem_id'],
            'name':spec['name'],'domain':spec['domain'],
            'capabilities':public_capabilities(description),'state_slots':fields,
            'lifecycle':lifecycle.manifest()})
    payload={'version':'public-catalog-v1','source_hashes':sources(),
             'scope':'Public implemented native contracts; visibility is not complete task acceptance.',
             'devices':records}
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=True,allow_nan=False)+'\n')
    return payload


def load_catalog(path=CACHE):
    data=json.loads(path.read_text())
    if data['source_hashes']!=sources():raise ValueError('Public catalog is stale; rebuild offline')
    ids=[d['id'] for d in data['devices']]
    if len(ids)!=len(set(ids)) or set(ids)!={s['slug'] for s in load_registry()['devices']}:
        raise ValueError('Public catalog does not cover the registry exactly')
    return data


def index(catalog):
    return [{k:d[k] for k in ['id','name','domain']} for d in catalog['devices']]


def search(catalog,query='',offset=0,limit=30):
    if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=50:
        raise ValueError('Invalid public catalog page')
    terms=set(re.findall(r'[a-z0-9]+',query.lower()))
    rows=index(catalog)
    if terms:
        rows=[d for d in rows if terms & set(re.findall(r'[a-z0-9]+',json.dumps(d).lower()))]
    return {'total':len(rows),'offset':offset,'items':rows[offset:offset+limit],
            'next_offset':offset+limit if offset+limit<len(rows) else None}


def describe(catalog,device_id):
    result=next((d for d in catalog['devices'] if d['id']==device_id),None)
    if result is None:raise ValueError('Unknown registered device ID')
    return deepcopy(result)
