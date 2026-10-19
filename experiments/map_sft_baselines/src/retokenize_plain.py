"""Encode the same accepted routes for the no-map 7B R+A baseline."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import torch
from transformers import AutoTokenizer
from grounded_llm.tasks.blocks.sft import demonstration, atomic_json
from grounded_llm.core.sft.transcript import encode_trajectory


def initialize(source, out, model, maximum):
    global SOURCE, OUT, TOKENIZER, MAXIMUM
    SOURCE, OUT, MAXIMUM = Path(source), Path(out), maximum
    TOKENIZER = AutoTokenizer.from_pretrained(model, local_files_only=True)
    torch.set_num_threads(1)


def convert(row):
    packed = torch.load(SOURCE/'trajectories'/row['prepared_file'], map_location='cpu', weights_only=False)
    steps = packed['map']['steps']
    demo = demonstration(row, None, False, cached_steps=steps)
    encoded = encode_trajectory(demo, TOKENIZER, MAXIMUM)
    torch.save(dict(plain=dict(encoded=encoded, steps=steps)), OUT/'trajectories'/row['prepared_file'])
    return dict(row, plain_tokens=len(encoded.input_ids))


def main():
    parser = argparse.ArgumentParser()
    for name in ['source','out','model-path','config']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    (args.out/'trajectories').mkdir(parents=True)
    meta = json.loads((args.source/'dataset.json').read_text())
    cfg = json.loads(args.config.read_text())
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                             initializer=initialize,initargs=(str(args.source),str(args.out),str(args.model_path),cfg['maximum_sequence_tokens'])) as pool:
        for row in pool.map(convert,meta['rows'],chunksize=16):
            rows.append(row)
            if len(rows)%500==0:
                atomic_json(args.out/'progress.json',dict(samples=len(rows),total=len(meta['rows'])))
                print(json.dumps(dict(samples=len(rows))),flush=True)
    if len(rows)!=cfg['data']['samples_per_epoch']:
        raise ValueError('Incomplete subset')
    atomic_json(args.out/'dataset.json',dict(meta,config=cfg,rows=rows,response_order='reason_first',source_data=str(args.source),tokenizer=str(args.model_path)))
    atomic_json(args.out/'ready.json',dict(samples=len(rows),smoke=False,response_order='reason_first',max_tokens=max(r['plain_tokens'] for r in rows)))


if __name__=='__main__':
    main()
