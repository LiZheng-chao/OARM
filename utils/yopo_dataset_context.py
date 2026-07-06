import os
from contextlib import contextmanager

from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


def resolve_dataset_dir(dataset_root=None):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    return os.path.abspath(dataset_root or os.path.join(repo_root, 'dataset'))


@contextmanager
def yopo_dataset_cfg(dataset_root=None):
    dataset_dir = resolve_dataset_dir(dataset_root)
    old_dataset_path = cfg['dataset_path']
    cfg['dataset_path'] = dataset_dir
    try:
        yield dataset_dir
    finally:
        cfg['dataset_path'] = old_dataset_path
