"""Download official weights with optional local staging for slow shared storage."""
import argparse
import json
import os
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-id', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--staging-dir', required=True)
    p.add_argument('--revision', default='master')
    p.add_argument('--parallel-workers', type=int, default=8)
    args = p.parse_args()
    os.environ['MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS'] = str(args.parallel_workers)
    from modelscope import snapshot_download
    staging = Path(args.staging_dir).resolve()
    destination = Path(args.model_dir).resolve()
    staging.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(args.model_id, revision=args.revision, local_dir=str(staging), max_workers=4)
    if staging != destination:
        for source in staging.iterdir():
            if source.is_file() and (source.suffix in ('.json', '.safetensors', '.txt', '.md')
                    or source.name in ('LICENSE', '.gitattributes', '.msc', '.mdl', '.mv')):
                shutil.copy2(source, destination / source.name)
    (destination / 'grounded_download_complete.json').write_text(json.dumps({
        'model_id': args.model_id, 'revision': args.revision, 'provider': 'ModelScope',
        'model_dir': str(destination), 'staging_dir': str(staging)}, indent=2))
    print(str(destination), flush=True)


if __name__ == '__main__':
    main()
