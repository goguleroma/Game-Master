from simulated_game.benchmark import run_benchmark


def test_benchmark_reports_matched_outcomes_and_failure_metrics() -> None:
    report = run_benchmark(
        scenario="static",
        episodes=2,
        max_turns=1,
        sequences=1,
        order_seed=7,
    )
    sequence = report["sequences"][0]

    assert sequence["volatile"]["episodes"] == 2
    assert sequence["persistent"]["episodes"] == 2
    assert sequence["volatile"]["timeouts"] == 2
    assert sequence["persistent"]["timeouts"] == 2
    assert "blocked_by_boundary" in sequence["volatile"]["action_status_counts"]
    assert "prediction_accuracy" in sequence["persistent"]
    assert sequence["volatile"]["memory_retrieval"]["queries"] == 2
    assert "guided_decisions" in sequence["persistent"]["memory_retrieval"]
