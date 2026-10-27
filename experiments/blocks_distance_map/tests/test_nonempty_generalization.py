import unittest
import numpy as np
import torch
from torch.nn import functional as F

from grounded_llm.core.qmap.model import soft_rank_loss
from grounded_llm.tasks.blocks.model import BoardEncoder, board_bits
from experiments.blocks_distance_map.src.full_graph_run import relation_batch
from experiments.blocks_distance_map.src.nonempty_generalization import star_losses


class NonemptyChecks(unittest.TestCase):
    def test_static_full_stars_match_original_joint_loss_and_gradient(self):
        torch.set_num_threads(2)
        masks=[0,3,12,15,48,63];features=board_bits(masks)
        data=dict(parents=torch.tensor([5,3]),goals=torch.tensor([0,1]),
            children=torch.tensor([[3,4,2],[2,1,-1]]),truth=torch.tensor([[2.,1.,-1.],[-1.,0.,-1.]]),
            parent_truth=torch.tensor([3.,1.]))
        ids=torch.tensor([1,0,1]);opt=dict(unreachable_label=19,temperature=1)
        model=BoardEncoder(128,16,13,'directed_sum')
        h,r=star_losses(model,features,data,ids,opt);actual=h+r
        ga=torch.autograd.grad(actual,tuple(model.parameters()))
        src=[];goal=[];truth=[];rs=[];rg=[];rt=[];groups=[]
        for group,i in enumerate(ids.tolist()):
            s,g=int(data['parents'][i]),int(data['goals'][i]);src.append(masks[s]);goal.append(masks[g]);truth.append(float(data['parent_truth'][i]))
            for c,t in zip(data['children'][i].tolist(),data['truth'][i].tolist()):
                if c<0:continue
                src.append(masks[c]);goal.append(masks[g]);truth.append(t)
                rs.append(masks[c]);rg.append(masks[g]);rt.append(t);groups.append(group)
        pack=lambda a:np.asarray([[v,0] for v in a],dtype=np.uint64)
        batch=(pack(src),pack(goal),np.asarray(truth),pack(rs),pack(rg),np.asarray(rt),np.asarray(groups))
        bits,pairs,labels,target,order,slots,shape=relation_batch(batch,'cpu',19)
        prediction=model(bits,pairs);scores=torch.zeros(np.prod(shape)).scatter(0,slots,prediction[len(labels):][order]).reshape(shape)
        labels=labels.float();target=target.float();desired=torch.zeros(np.prod(shape)).scatter(0,slots,target).reshape(shape)
        valid=torch.zeros(np.prod(shape),dtype=torch.bool);valid[slots]=True
        expected=F.smooth_l1_loss(prediction[:len(labels)],labels)+soft_rank_loss(scores,desired,valid.reshape(shape),1)
        gb=torch.autograd.grad(expected,tuple(model.parameters()))
        torch.testing.assert_close(actual,expected)
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()
