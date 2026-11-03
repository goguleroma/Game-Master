import subprocess

print("Run 1:")
result = subprocess.run(
    ["python", "-m", "pytest", "tests/test_simulated_game.py::test_agent_outcomes_with_active_pursuer_across_multiple_seeds", "-v", "--tb=short"],
    cwd="C:\\Users\\0102\\Documents\\GitHub\\Game-Master",
    capture_output=True,
    text=True,
    timeout=180
)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print(f"Return code: {result.returncode}")