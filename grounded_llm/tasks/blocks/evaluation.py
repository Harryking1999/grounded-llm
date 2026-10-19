"""Score a generated board action against physical rules and map ranking."""
from .protocol import parse_control
from grounded_llm.core.sft.ranking import parse_ranking, meeting_ranking_scores

def score(text,step,oracle,with_map):
    before=oracle.distance(step.current,step.goal);candidate,error=parse_control(text,len(step.candidate_actions))
    row=dict(current=str(step.current),goal=str(step.goal),answer=text,candidate=candidate,error=error,live=before>0,rank_eligible=not step.done and bool(step.candidate_actions),safe=False,optimal=False,map_optimal=False)
    if candidate is not None and error is None:
        _,after=step.execute(candidate);remaining=oracle.distance(after,step.goal)
        row.update(after=str(after),safe=before>0 and remaining>=0,optimal=before>0 and remaining==before-1,map_optimal=candidate in step.map_minimal_candidates)
    if with_map and row['rank_eligible']:
        row.update(meeting_ranking_scores(step.candidate_map_distances,parse_ranking(text,len(step.candidate_actions),10)))
    return row
