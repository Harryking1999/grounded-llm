"""Short map-comparison question and exact-answer scoring."""

import re


def diagnostic_prompt(user_text, mode, candidate_pair):
    if mode == "pairwise":
        left, right = candidate_pair
        question = (f"Between candidate {left} and candidate {right}, which successor "
                    "is closer to the goal in the supplied Q-map?")
        tag = "closer"
    elif mode == "nearest":
        question = "Which legal candidate has the smallest map distance to the goal?"
        tag = "nearest"
    else:
        raise ValueError(mode)
    return (f"{user_text}\n\n[Map readout diagnostic]\n{question} "
            f"Reply only <{tag}>candidate_ID</{tag}>."), tag


def parse_short_answer(answer, tag, candidate_count):
    match = re.fullmatch(rf"\s*<{tag}>\s*([1-9][0-9]*)\s*</{tag}>\s*", answer)
    if match is None:
        return None
    chosen = int(match.group(1))
    return chosen if chosen <= candidate_count else None


def correct_ids(step, mode, pair):
    if mode == "nearest":
        return set(step.map_minimal_candidates)
    left, right = pair
    distances = step.candidate_map_distances
    return {left if distances[left - 1] < distances[right - 1] else right}
