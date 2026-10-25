import unittest
import torch
from grounded_llm.core.qmap.model import StateEncoder,widen_mlp_weights


class WideningTests(unittest.TestCase):
    def test_initial_function_matches_without_permanent_clone_symmetry(self):
        torch.manual_seed(7);old=StateEncoder(16,12,64,'linf',input_dim=10)
        x=torch.randn(128,10)
        for factor in (4,8):
            new=StateEncoder(16,12*factor,64,'linf',input_dim=10)
            new.load_state_dict(widen_mlp_weights(old.state_dict(),factor,19))
            torch.testing.assert_close(new.encode(x),old.encode(x),rtol=0,atol=1e-4)
            self.assertFalse(torch.equal(new.network[4].weight[:,0],new.network[4].weight[:,1]))
            loss=new.encode(x).square().mean();loss.backward()
            self.assertFalse(torch.equal(new.network[2].weight.grad[0],new.network[2].weight.grad[1]))


if __name__=='__main__':unittest.main()
