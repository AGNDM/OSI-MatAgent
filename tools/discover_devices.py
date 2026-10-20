"""Inventory pinned runtime implementations; never download or call a model."""
import inspect
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.device_registry import VENDOR_ROOT, open_native, public_capabilities


def discover():
    catalog = json.loads((VENDOR_ROOT / 'catalog/catalog.json').read_text())['assets']
    rows, failures = [], []
    classes = {}
    # The pinned catalog omits runtime paths for these exact IDs; the official
    # implementations declare matching ASSET_TYPE constants in these modules.
    missing_paths = {
        'magritek_spinsolve80':'basic40_runtime_measure',
        'agilent_hplc1260':'basic40_runtime_analytical',
        'mt_quantos_q2_xpe206dr':'basic40_runtime_automation',
        'opentrons_flex_single_5_1000ul':'basic40_runtime_automation',
    }
    for item in catalog:
        module = item['files'].get('runtime', '').removeprefix('scripts/').removesuffix('.py').replace('/', '.')
        module = module or missing_paths.get(item['slug'], '')
        base = {k: item[k] for k in ['slug', 'chem_id', 'name', 'name_zh', 'domain']}
        base.update(asset_id=item['id'], module=module)
        try:
            m = importlib.import_module(module)
            if module not in classes:
                classes[module] = [name for name, value in vars(m).items() if inspect.isclass(value)
                    and value.__module__ == module and hasattr(value, 'register_principal') and hasattr(value, 'session')]
            candidates = classes[module]
            errors = []
            for cls in candidates:
                spec = dict(base, **{'class': cls})
                try:
                    host, session, instance, desc = open_native(spec)
                    actual = desc.get('asset_type_id')
                    if actual and actual != item['id']:
                        continue
                    # Single-runtime modules can omit the ID; shared hosts must identify it.
                    if not actual and len(candidates) > 1:
                        sig = inspect.signature(host.create_instance) if hasattr(host, 'create_instance') else None
                        if not sig or sig.parameters.get('asset_type_id') is None:
                            continue
                    observation = session.read(instance_id=instance)
                    caps = public_capabilities(desc)
                    rows.append(dict(spec, capability_count=len(caps), slot_count=len(observation.get('observations', {}))))
                    break
                except FileNotFoundError:
                    raise
                except Exception as e:
                    errors.append(f'{cls}: {type(e).__name__}: {e}')
            else:
                raise ValueError('; '.join(errors) or 'No matching native host')
        except Exception as e:
            failure = dict(base, error=f'{type(e).__name__}: {e}')
            if isinstance(e, FileNotFoundError):
                failure['missing'] = e.filename
            failures.append(failure)
    return {'code_revision': '123cf78593469871e41b8dbd63a11dfb7c5ca251',
        'catalog_count': len(catalog), 'devices': rows, 'unavailable': failures}


if __name__ == '__main__':
    data = discover()
    path = ROOT / 'cases/devices/inventory.json'
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(f"Native sessions: {len(data['devices'])}/{data['catalog_count']}; unavailable: {len(data['unavailable'])}")
    for r in data['unavailable']:
        print(r['slug'], r['error'][:180])
