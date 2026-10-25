"""Configure the blocks study and invoke the shared SFT trainer."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from grounded_llm.tasks.blocks.dataset import Routes
from grounded_llm.core.sft.training import train, shard_model, save, save_sharded
from grounded_llm.core.sft.schedule import learning_rate_factor, validate_resume_contract


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'data', 'model-path', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--extend-training', action='store_true')
    parser.add_argument('--reinitialize-optimizer', action='store_true',
                        help='Load resume weights; estimate fresh Adam moments using configured calibration batches')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    spec = config['baselines'][args.baseline]
    dataset_class = Routes
    if config.get('task') == 'multitask':
        from .multitask_data import Routes as MultitaskRoutes
        dataset_class = MultitaskRoutes
    if config.get('task') in ('cube2', 'puzzle8'):
        from grounded_llm.tasks.restoration_sft import Routes as RestorationRoutes
        dataset_class = RestorationRoutes
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    data = dataset_class(args.data, spec['real_map'], config=config,
                  baseline=args.baseline, tokenizer=tokenizer)
    if spec['real_map'] and data.meta.get('response_order', 'reason_first') != spec.get('response_order', 'reason_first'):
        raise ValueError('Prepared assistant order differs from baseline configuration')
    ready = json.loads((args.data / 'ready.json').read_text())
    if not args.smoke_steps and (len(data) != config['data']['samples_per_epoch'] or ready['smoke']):
        raise ValueError('Formal training requires accepted complete common data')
    train(config, args.baseline, args.model_path, data, args.out,
          pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
          smoke_steps=args.smoke_steps, resume=args.resume, extend_training=args.extend_training,
          reinitialize_optimizer=args.reinitialize_optimizer)


if __name__ == '__main__':
    main()
