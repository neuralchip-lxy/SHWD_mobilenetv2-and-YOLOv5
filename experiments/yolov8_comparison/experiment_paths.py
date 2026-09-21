"""Repository-relative defaults; optional ignored machine-local overrides.

CLI/config paths are relative to REPO_ROOT, never the shell's working directory.
Dataset YAML relative roots are relative to that YAML, not Ultralytics settings.
"""
import json
import hashlib
import os
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    'runs': 'outputs/analysis_runs',
    'shwd_yaml': 'SHWD_YOLO/shwd.yaml',
    'shwd_root': 'SHWD_YOLO',
    'shel5k_yolo': 'datasets/SHEL5K6_YOLO_v1',
    'shel5k_source': 'datasets/SHEL5K_original/9rcv8mm682-4/Safety Helmet Wearing Dataset',
    'shel5k_audit': 'outputs/shel5k_preparation',
}


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def settings():
    config = resolve_path(os.environ.get('M2Y5_PATHS_CONFIG', 'experiment_paths.local.json'))
    values = dict(DEFAULTS)
    if config.is_file():
        override = json.loads(config.read_text(encoding='utf-8-sig'))
        unknown = set(override) - set(values)
        if unknown:
            raise ValueError(f'Unknown keys in {config}: {sorted(unknown)}')
        values.update(override)
    elif 'M2Y5_PATHS_CONFIG' in os.environ:
        raise FileNotFoundError(config)
    return values


def path_for(key):
    return resolve_path(settings()[key])


def normalize_data_yaml(value, root_override=None):
    source = resolve_path(value)
    if not source.is_file():
        return source  # let entry point report the missing dataset
    data = yaml.safe_load(source.read_text(encoding='utf-8'))
    old_root = Path(data.get('path', '.'))
    root = resolve_path(root_override) if root_override is not None else (
        old_root.resolve() if old_root.is_absolute() else (source.parent / old_root).resolve())
    if old_root.is_absolute() and old_root.resolve() == root:
        return source  # preserve original YAML bytes and checkpoint identity
    data['path'] = root.as_posix()
    content = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    cache = path_for('runs') / '_resolved_data'
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (source.stem + '_' + hashlib.sha256(content.encode()).hexdigest()[:16] + '.yaml')
    if not target.exists():
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=cache, delete=False) as f:
            f.write(content)
            temp = Path(f.name)
        temp.replace(target)
    return target


def shwd_data():
    return normalize_data_yaml(path_for('shwd_yaml'), path_for('shwd_root'))


def shel5k_data():
    root = path_for('shel5k_yolo')
    return normalize_data_yaml(root / 'shel5k.yaml', root)
