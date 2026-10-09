"""A copied adapter may change locations, never its supervision configuration."""
import unittest
from experiments.flamingo_map_reader.src.artifact_paths import relocate_supervision_paths


class ArtifactPathsTest(unittest.TestCase):
    def test_only_explicit_locations_change(self):
        source = dict(manifest='/platform/data/manifest.json', model_source='/platform/model',
                      map_source='/platform/qmap.pt', config=dict(seed=7, training=dict(epochs=10)))
        mappings = {field:dict(source=source[field], destination=source[field].replace('/platform','/target'))
                    for field in ('manifest','model_source','map_source')}
        migrated = relocate_supervision_paths(source, mappings)
        self.assertEqual(migrated['config'], source['config'])
        self.assertEqual(source['model_source'], '/platform/model')
        self.assertEqual(migrated['model_source'], '/target/model')
        with self.assertRaises(ValueError):
            relocate_supervision_paths(source, dict(seed=dict(source=7,destination=8)))
        with self.assertRaises(ValueError):
            relocate_supervision_paths(source, dict(model_source=dict(source='foreign',destination='/target/model')))
