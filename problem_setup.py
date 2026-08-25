from desdeo.problem.testproblems import river_pollution_problem


def create_problem():
    """Create the four-objective river pollution problem."""
    return river_pollution_problem(
        five_objective_variant=False,
    )
