"""Compatibility imports; implementation lives in grounded_llm.tasks.blocks.protocol."""
from grounded_llm.tasks.blocks.protocol import (
    SimpleNamespace, rules, DistanceOracle, initial_prompt, turn_prompt, atomic_json, RESPONSE_INSTRUCTION, STOP_REASON, CONTINUE_SUFFIX, ARRIVAL_SUFFIX, legal_actions, action_key, action_from_key, oracle_for, sample_routes, reason, response, user_update, route_messages, encode_messages, parse_control, check_reason
)
