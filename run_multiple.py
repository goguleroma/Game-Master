import subprocess

for i in range(5):
    print(f"Run {i+1}:")
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/test_simulated_game.py::test_agent_outcomes_with_active_pursuer_across_multiple_seeds", "-v", "--tb=short"],
        cwd="C:\\Users\\0102\\Documents\\GitHub\\Game-Master",
        capture_output=True,
        text=True
    )
    # Print last 10 lines
    lines = result.stdout.strip().split('\n')
    for line in lines[-10:]:
        print(f"  {line}")
    if result.returncode != 0:
        print("  FAILED")
    else:
        print("  PASSED")
    print()