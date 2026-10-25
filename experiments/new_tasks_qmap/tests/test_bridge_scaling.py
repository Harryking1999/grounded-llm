import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from grounded_llm.core.qmap.model import StateEncoder
from grounded_llm.tasks.bridge.environment import Bridge
from grounded_llm.tasks.bridge.qmap import MapReader


class ScalingTests(unittest.TestCase):
    def test_physics_features_and_model_scale_together(self):
        torch.manual_seed(71)
        env=Bridge([1,2,4,7,11,17],'time_homogeneous');scaled=Bridge([3*t for t in env.times],'time_homogeneous')
        states=np.arange(128);features=env.features(states)
        np.testing.assert_array_equal(scaled.features(states),3*features)
        graph=env.graph(43);other=scaled.graph(43)
        np.testing.assert_array_equal(graph['children'],other['children'])
        np.testing.assert_array_equal(other['costs'],3*graph['costs'])
        np.testing.assert_array_equal(other['distances'],3*graph['distances'])
        model=StateEncoder(dimension=128,hidden=64,cap=256,metric='linf',input_dim=13,bias=False,output_activation='linear')
        with torch.no_grad():
            a=model.encode(torch.from_numpy(features));b=model.encode(torch.from_numpy(scaled.features(states)))
        torch.testing.assert_close(b,3*a,rtol=2e-5,atol=1e-4)
        torch.testing.assert_close(model.distance(b[:64],b[64:]),3*model.distance(a[:64],a[64:]),rtol=2e-5,atol=1e-4)

    def test_reader_uses_saved_encoding_and_old_default_still_loads(self):
        env=Bridge([1,2,4,7,11,17]);states=[0,43,127]
        with tempfile.TemporaryDirectory() as tmp:
            for encoding in ['identity','time_homogeneous']:
                model_cfg=dict(dimension=8,hidden=16,cap=256,metric='linf')
                if encoding!='identity':model_cfg.update(bias=False,output_activation='linear')
                cfg=dict(model=model_cfg)
                if encoding!='identity':cfg['input_encoding']=encoding
                model=StateEncoder(input_dim=13,**model_cfg)
                path=Path(tmp)/f'{encoding}.pt'
                torch.save(dict(model_type='exact_state_pairs',task='bridge',model=model.state_dict(),config=cfg,
                    scope=dict(people=6),step=0,input_dim=13),path)
                with patch.object(Bridge,'graph',side_effect=RuntimeError('No oracle at inference')):
                    reader=MapReader(env,path)
                    expected=model.encode(torch.from_numpy(Bridge(env.times,encoding).features(states)))
                    torch.testing.assert_close(reader.encode(states),expected)


if __name__=='__main__':unittest.main()
