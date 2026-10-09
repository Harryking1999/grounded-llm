"""Explicit relocation of copied artifacts without changing supervision."""


def relocate_supervision_paths(contract, path_map):
    relocated = dict(contract)
    allowed = {'manifest', 'map_source', 'model_source'}
    if set(path_map) - allowed:
        raise ValueError('Only artifact locations can be relocated')
    for field, mapping in path_map.items():
        if set(mapping) != {'source', 'destination'}:
            raise ValueError('Artifact relocation requires source and destination')
        if contract.get(field) != mapping['source']:
            raise ValueError(f'Unexpected source artifact for {field}')
        relocated[field] = mapping['destination']
    return relocated
