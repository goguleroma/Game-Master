from simulated_game.benchmark import run_benchmark


def test_benchmark_reports_matched_outcomes_and_failure_metrics() -> None:
    report = run_benchmark(
        scenario="static",
        episodes=2,
        max_turns=10,
        sequences=1,
        order_seed=7,
    )
    sequence = report["sequences"][0]

    assert sequence["volatile"]["episodes"] == 2
    assert sequence["persistent"]["episodes"] == 2
    # With PathTree planner, agent makes smarter moves toward target
    # rather than blindly hitting boundaries. Verify action was taken.
    assert len(sequence["volatile"]["action_status_counts"]) > 0
    assert "prediction_accuracy" in sequence["persistent"]
    # PathTreePlanner (MCTS) uses retrieval extensively during tree search:
    # at root, during expansion, and during simulation rollouts.
    # This results in many more queries than the BFS planner.
    assert sequence["volatile"]["memory_retrieval"]["queries"] > 2
    assert "guided_decisions" in sequence["persistent"]["memory_retrieval"]
    # Both volatile and persistent should produce valid reports
    assert "mean_turns" in sequence["volatile"]
    assert "mean_turns" in sequence["persistent"]
