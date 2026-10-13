"""Re-encode existing trajectories in a new response order without resampling maps."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import torch
from transformers import AutoTokenizer
from .common import demonstration, atomic_json
from experiments.flamingo_map_reader.src.transcript import encode_trajectory


def initialize(source, destination, model_path, order, maximum):
    global SOURCE, DESTINATION, TOKENIZER, ORDER, MAXIMUM
    SOURCE, DESTINATION = Path(source), Path(destination)
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    ORDER, MAXIMUM = order, maximum
    torch.set_num_threads(1)


def convert(row):
    packed = torch.load(SOURCE/'trajectories'/row['prepared_file'], map_location='cpu', weights_only=False)
    demo = demonstration(row, None, True, ORDER, packed['map']['steps'])
    encoded = encode_trajectory(demo, TOKENIZER, MAXIMUM)
    packed['map']['encoded'] = encoded
    torch.save(packed, DESTINATION/'trajectories'/row['prepared_file'])
    return dict(row, map_tokens=len(encoded.input_ids))


def main():
    parser = argparse.ArgumentParser()
    for name in ('source','out','model-path','config'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--order',choices=['ranking_first'],default='ranking_first')
    args = parser.parse_args()
    if args.out.resolve()==args.source.resolve() or args.out.exists():
        raise ValueError('Use a fresh destination; never overwrite an existing training cache')
    ready=json.loads((args.source/'ready.json').read_text())
    if ready['smoke']:raise ValueError('Source data must be complete')
    data=json.loads((args.source/'dataset.json').read_text())
    config=json.loads(args.config.read_text())
    (args.out/'trajectories').mkdir(parents=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(str(args.source),str(args.out),str(args.model_path),args.order,config['maximum_sequence_tokens'])) as pool:
        for row in pool.map(convert,data['rows'],chunksize=32):
            rows.append(row)
            if len(rows)%1000==0:
                atomic_json(args.out/'progress.json',dict(samples=len(rows),total=len(data['rows'])))
                print(json.dumps(dict(samples=len(rows))),flush=True)
    for name in ('evaluation.json','probe_indices.json'):
        shutil.copyfile(args.source/name,args.out/name)
    atomic_json(args.out/'dataset.json',dict(data,rows=rows,response_order=args.order,source_data=str(args.source)))
    atomic_json(args.out/'ready.json',dict(ready,response_order=args.order))


if __name__=='__main__':main()
