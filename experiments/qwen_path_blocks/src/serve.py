"""Launch a local SGLang worker with runtime paths outside the research contract."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--cache-dir', required=True)
    args = parser.parse_args()
    serving = json.loads(Path(args.config).read_text())['serving']
    cache = Path(args.cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''),
               CUDA_VISIBLE_DEVICES=args.gpus,
               HF_HOME=str(cache / 'huggingface'), TRITON_CACHE_DIR=str(cache / 'triton'),
               TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'))
    command = [sys.executable, '-m', 'sglang.launch_server', '--model-path', args.model_path,
               '--host', '127.0.0.1', '--port', str(args.port),
               '--tp-size', str(len(args.gpus.split(','))), '--dtype', serving['dtype'],
               '--context-length', str(serving['context_length']),
               '--mem-fraction-static', str(serving['mem_fraction_static']),
               '--max-running-requests', str(serving['max_running_requests'])]
    print(json.dumps({'command': command, 'gpus': args.gpus}), flush=True)
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == '__main__':
    main()
