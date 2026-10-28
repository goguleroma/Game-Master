import sys

def main() -> None:
    print(
        "No environment is configured. Provide a decision policy, action executor, "
        "and, when needed, an observation adapter when creating GameEngine.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
