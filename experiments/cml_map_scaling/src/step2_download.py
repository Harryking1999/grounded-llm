"""Download from ModelScope and bind it to the contract's upstream revision."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen


def digest(path, kind):
    checksum = hashlib.new(kind)
    if kind == 'sha1':
        checksum.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/cml_map_scaling/configs/step2.json')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--upstream-metadata', help='Optional response from the exact HF revision API')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    spec = config['model']
    assert spec['revision'] == spec['tokenizer_revision']
    api = f"https://huggingface.co/api/models/{spec['id']}/revision/{spec['revision']}?blobs=true"
    if args.upstream_metadata:
        upstream = json.loads(Path(args.upstream_metadata).read_text(encoding='utf-8'))
    else:
        with urlopen(api, timeout=60) as response:
            upstream = json.load(response)
    assert upstream['sha'] == spec['revision'] and upstream['id'] == spec['id']
    from modelscope import snapshot_download
    destination = Path(args.model_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(spec['id'], revision=spec['modelscope_revision'], local_dir=str(destination), max_workers=4)
    checked = []
    for entry in upstream['siblings']:
        name = entry['rfilename']
        if not name.endswith(('.json', '.safetensors', '.txt')):
            continue
        path = destination / name
        assert path.stat().st_size == entry['size'], name
        if 'lfs' in entry:
            expected, kind = entry['lfs']['sha256'], 'sha256'
        else:
            expected, kind = entry['blobId'], 'sha1'
        if digest(path, kind) != expected:
            raise ValueError(f'ModelScope file differs from locked upstream revision: {name}')
        checked.append(name)
        print('Verified', name, flush=True)
    provenance = dict(model_id=spec['id'], provider='ModelScope',
                      modelscope_revision=spec['modelscope_revision'], verified_hf_revision=spec['revision'],
                      verification='Runtime files matched upstream LFS SHA256 or Git blob IDs; provider commit IDs differ.',
                      checked_files=checked, upstream_metadata_url=api,
                      completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (destination / 'step2_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
