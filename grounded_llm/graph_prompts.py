"""Text rendering for graph questions; wording remains in the run config."""

import numpy as np


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
