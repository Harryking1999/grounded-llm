import copy
import random

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.llm_task_sft.src.protocol import (DistanceOracle, check_reason, encode_messages,
    legal_actions, parse_control, response, route_messages, rules, sample_routes)
from experiments.llm_task_sft.src.evaluate import evaluate_task, score_turn, summarize
from experiments.llm_task_sft.src.training import SupervisedLM


def oracle_config():
    return dict(data=dict(oracle_cache_limit=100000,oracle_seconds_per_task=30),evaluation=dict(action_budget=32))


def test_shortest_routes_and_replacement():
    goal = 1 << 99
    start = goal | sum(1<<i for i in (0,1,2,10,11,12))
    oracle = DistanceOracle()
    rows, unique, minimum = sample_routes(start,goal,7,random.Random(1),oracle)
    assert unique == 6 and minimum == 2 and sum(repeat for _,repeat in rows) == 1
    assert len({route for route,_ in rows}) == 6
    for route,_ in rows:
        messages,_ = route_messages(start,goal,route,oracle)
        state = start
        for index,key in enumerate(route):
            candidates = legal_actions(state)
            chosen,error = parse_control(messages[index*2+1]['content'],len(candidates))
            assert error is None and check_reason(messages[index*2+1]['content'],state,goal,chosen,oracle) is None
            state = rules.apply(state,candidates[chosen-1])
        assert state == goal and messages[-1]['content'] == response(goal,goal)


def test_reason_failure_does_not_replace_control_and_wrong_stop_keeps_arrival():
    goal, start = 1<<99, (1<<99)|15
    oracle = DistanceOracle()
    text = 'Reason: incorrect\n<action>1</action>'
    judged = score_turn(text,start,goal,oracle)
    assert judged['reason_error']=='format' and judged['control_error'] is None and judged['shortest_progress']
    for text in ('<action>1</action>\n<action>1</action>','Reason: x\n<action>1</action> trailing','Reason: x\n<action>0</action>',
                 '<action>broken\n<action>1</action>'):
        assert parse_control(text,5)[1] == 'control_format'
    route = [(5,0,0),(5,0,2)]
    record=dict(start=str(start),goal=str(goal),reference_actions=route,trajectory_id='tiny',split='validation',
                sft_initial_pair_exposed=False,reference_nonterminal_pairs_exposed=0)
    replies=iter(['bad reason\n<action>1</action>','bad reason\n<action>1</action>','bad stop'])
    case=evaluate_task(record,'rollout',lambda _:dict(text=next(replies)),oracle_config())
    assert case['arrived'] and not case['arrived_and_stopped'] and case['failure']=='control_format'
    metrics=summarize([case]); assert metrics['arrival']['rate']==1 and metrics['arrival_and_stop']['rate']==0
    ref=summarize([dict(case,mode='reference')]); assert ref['arrival'] is None


def test_chunked_weighted_loss_and_gradients_match_full_vocabulary():
    torch.manual_seed(4)
    config=Qwen2Config(vocab_size=40,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                      num_attention_heads=2,num_key_value_heads=2,tie_word_embeddings=True,attention_dropout=0.)
    base=Qwen2ForCausalLM(config); expected=copy.deepcopy(base)
    actual=SupervisedLM(base,chunk_tokens=2).train()
    ids=torch.randint(0,40,(2,9)); mask=torch.ones_like(ids); labels=ids.clone()
    labels[:,:3]=-100; labels[0,6:]=-100
    weights=torch.tensor([.8,2.4])
    loss=actual(ids,mask,labels,weights)
    logits=expected(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,:-1].float()
    per_token=torch.nn.functional.cross_entropy(logits.reshape(-1,40),labels[:,1:].reshape(-1),reduction='none').reshape(2,-1)
    reference=(per_token.sum(1)/labels[:,1:].ne(-100).sum(1)*weights).mean()
    torch.testing.assert_close(loss,reference)
    loss.backward(); reference.backward()
    for a,b in zip(base.parameters(),expected.parameters()):
        assert a.requires_grad
        torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-4)


def test_real_qwen_template_masks_only_answers():
    from pathlib import Path
    from transformers import AutoTokenizer
    from experiments.flamingo_map_reader.src.text import chat_ids
    path=Path('tmp/platform-twenty-startup/transfer/models/Qwen2.5-1.5B-Instruct')
    if not path.exists(): pytest.skip('Local Qwen tokenizer not available')
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    start,goal=(1<<99)|15,1<<99
    oracle=DistanceOracle()
    rows,_,_=sample_routes(start,goal,1,random.Random(1),oracle)
    messages,_=route_messages(start,goal,rows[0][0],oracle)
    ids,labels=encode_messages(messages,tokenizer)
    prefix,expected=[],[]
    for i in range(0,len(messages),2):
        generation=chat_ids(tokenizer,messages[:i+1],add_generation_prompt=True)
        complete=chat_ids(tokenizer,messages[:i+2],add_generation_prompt=False)
        assert complete[:len(generation)]==generation and generation[:len(prefix)]==prefix
        expected.extend([-100]*(len(generation)-len(prefix))); expected.extend(complete[len(generation):]); prefix=complete
    assert ids==prefix and labels==expected
