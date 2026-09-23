"""Authority for local, frozen model/tokenizer loading."""

def load_model(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = config['model']
    directory = spec['directory']
    if spec.get('revision') and not spec.get('provenance'):
        raise ValueError('A revision-locked model requires its existing provenance file')
    if spec.get('provenance'):
        from .artifacts import read_json
        evidence = read_json(spec['provenance'])
        if evidence.get('verified_hf_revision') != spec['revision']:
            raise ValueError('Model provenance does not match the configured revision')
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=getattr(torch, spec['dtype']),
        attn_implementation=spec.get('attention', 'sdpa'), local_files_only=True).to(spec.get('device', 'cuda'))
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer
