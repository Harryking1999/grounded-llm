"""Authority for graph task wording and message rendering; no model or judge."""
import numpy as np

INPUT_KEYS = ('graph_text', 'current_id_text', 'goal_id_text', 'current_vector', 'goal_vector')

def build_prompt(example, task, input_config, config, adjacency):
    """Render independent textual-ID/vector visibility, without access to labels.

    Prompt templates remain config data. historical augmentation wording can be
    reused exactly via prompt_style=augmentation; unsupported changes fail.
    """
    item = dict(example, task=task)
    if config.get('prompt_style', 'state') == 'augmentation':
        expected = dict(graph_text=True, current_id_text=True, goal_id_text=True,
                        current_vector=input_config['current_vector'], goal_vector=input_config['current_vector'])
        if input_config != expected:
            raise ValueError('Historical augmentation template requires full text and matched vector roles')
        return augmentation_messages(config, adjacency, item, input_config['current_vector'], config['assets']['node_count'])
    prompts, template = config['prompts'], item['template']
    values = {}
    for role, key in (('current', 'u'), ('goal', 'g')):
        parts = []
        if input_config[role + '_id_text']:
            parts.append(str(item[key]))
        if input_config[role + '_vector']:
            parts.append('<|cml_' + role + '_state|>')
        values[role + '_state'] = ' '.join(parts) if parts else '[not provided]'
    state = prompts['state_blocks'][template].format(**values)
    user = prompts['user'].format(max_node_id=config['assets']['node_count'] - 1,
        adjacency=adjacency if input_config['graph_text'] else '', state_block=state,
        question=prompts['questions'][template][task])
    return [{'role': 'system', 'content': prompts['system']}, {'role': 'user', 'content': user}]

def adjacency_text(adj):
    """Render the same ordered adjacency list for every matched condition."""
    return '\n'.join(f"{i}: {', '.join(map(str, np.flatnonzero(row)))}" for i, row in enumerate(adj))

def messages(config, adjacency, item, text_control=False):
    """Build report messages from config prompts and a directed question record."""
    p, template = config['prompts'], item['template']
    current = str(item['u']) if text_control else '<|cml_current_state|>'
    goal = str(item['g']) if text_control else '<|cml_goal_state|>'
    state = p['state_blocks'][template].format(current_state=current, goal_state=goal)
    user = p['user'].format(max_node_id=config['assets']['node_count'] - 1,
                            adjacency=adjacency, state_block=state,
                            question=p['questions'][template][item['task']])
    return [{'role': 'system', 'content': p['system']}, {'role': 'user', 'content': user}]

def augmentation_messages(spec, adjacency, item, with_roadmap, node_count=32):
    prompts = spec['prompts']
    block = prompts['roadmap_block'].format(current_slot='<|cml_current_state|>',
                                             goal_slot='<|cml_goal_state|>') if with_roadmap else ''
    user = prompts['user'].format(max_node_id=node_count - 1, adjacency=adjacency, u=item['u'], g=item['g'], roadmap_block=block)
    return [{'role': 'system', 'content': prompts['system']}, {'role': 'user', 'content': user}]

def decision_messages(item,condition,adjacency,reason=None):
    task=item['task'];latent=condition.endswith('_latent');roadmap=condition!='text'
    roles=['a','g'] if task.startswith('neighbor') else ['g','a','b']
    if latent:
        text='图为固定的无向无权图。以下向量表示该图中的节点。\n'
    else:
        text='图为固定的无向无权图，每步沿一条边移动。\n邻接表：\n'+adjacency+'\n'
        text+='\n'.join(f'{role.upper()} 节点编号：{item[role]}' for role in roles)+'\n'
    if roadmap:
        text+='\n节点的 roadmap 表示：\n'+'\n'.join(f'{role.upper()}：<|diag_{role}|>' for role in roles)+'\n'
    if task.startswith('neighbor'):
        text+='A 是当前位置，G 是目标。请选择一个到 G 的最短距离严格小于 A 的合法邻居。\n合法邻居候选：'+', '.join(item['options'])+'。\n'
        instruction='只输出一个候选节点编号，不要解释。'
    elif task=='distance_compare':
        text+='目标为 G。从 A 和 B 分别沿图中的边到达 G，哪个位置需要的最少步数更少？两者距离不同。\n'
        instruction='只输出 A 或 B，不要解释。'
    else:
        text+='目标为 G。已沿一条边从 A 移到 B，到 G 的最短距离变得更近、相同还是更远？\n'
        instruction='只输出 更近、相同 或 更远，不要解释。'
    if task=='neighbor_reason' and reason is None:
        instruction='请在有限篇幅内分析候选方向，不要展开冗长的逐节点搜索。随后会单独要求你给出最终节点编号。'
    conversation=[dict(role='system',content='根据提供的图与状态回答问题。'),dict(role='user',content=text+instruction)]
    if reason is not None:
        conversation += [dict(role='assistant',content=reason),dict(role='user',content='分析阶段结束。现在必须从候选 '+', '.join(item['options'])+' 中选择一个下一节点，只输出编号。')]
    return conversation
