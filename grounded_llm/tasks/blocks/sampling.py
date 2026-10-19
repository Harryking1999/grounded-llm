"""Balance finite and certified-unreachable board relations."""

def strata(pairs):
    """Half weight to finite distances, half to certified negatives."""
    finite = [(pairs[:, 2] == 1), (pairs[:, 2] == 2),
              ((pairs[:, 2] >= 3) & (pairs[:, 2] <= 4)), (pairs[:, 2] >= 5)]
    negative = [(pairs[:, 3] == 1), (pairs[:, 3] == 2)]
    families = [[mask.nonzero(as_tuple=True)[0] for mask in family if mask.any()]
                for family in (finite, negative)]
    present = sum(bool(family) for family in families)
    return [(ids, 1 / present / len(family)) for family in families for ids in family]
