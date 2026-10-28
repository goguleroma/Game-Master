import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


Genome = dict[str, float]
Evaluator = Callable[[Genome, Sequence[int]], Sequence[float]]


@dataclass(frozen=True)
class StrategyCandidateResult:
    genome: Genome
    scores: tuple[float, ...]


@dataclass(frozen=True)
class StrategyResult:
    baseline: Genome
    best: Genome
    baseline_scores: tuple[float, ...]
    best_scores: tuple[float, ...]
    leaderboard: tuple[StrategyCandidateResult, ...]
    evaluated: int


class StrategySearch:
    """Environment-neutral evolutionary search over bounded numeric parameters."""

    def __init__(
        self,
        bounds: Mapping[str, tuple[float, float]],
        seed: int = 0,
        integer_parameters: Sequence[str] = (),
    ) -> None:
        if not bounds:
            raise ValueError("At least one strategy parameter is required.")
        normalized_bounds = {
            name: (float(low), float(high))
            for name, (low, high) in bounds.items()
        }
        if any(
            not math.isfinite(low)
            or not math.isfinite(high)
            or low >= high
            for low, high in normalized_bounds.values()
        ):
            raise ValueError("Each parameter bound must have low < high.")
        if not set(integer_parameters).issubset(bounds):
            raise ValueError("Integer parameters must be present in bounds.")
        if any(
            not float(low).is_integer() or not float(high).is_integer()
            for name, (low, high) in normalized_bounds.items()
            if name in integer_parameters
        ):
            raise ValueError("Integer parameter bounds must be whole numbers.")
        self.bounds = normalized_bounds
        self.integer_parameters = frozenset(integer_parameters)
        self._rng = random.Random(seed)

    def search(
        self,
        baseline: Mapping[str, float],
        seeds: Sequence[int],
        evaluate: Evaluator,
        generations: int = 5,
        population_size: int = 8,
    ) -> StrategyResult:
        if generations < 1 or population_size < 2:
            raise ValueError("Search requires generations >= 1 and population_size >= 2.")
        if not seeds:
            raise ValueError("At least one evaluation seed is required.")
        baseline_genome = self._validated(baseline)
        scores_by_genome: dict[tuple[tuple[str, float], ...], tuple[float, ...]] = {}

        def score(genome: Genome) -> tuple[float, ...]:
            key = self._key(genome)
            if key not in scores_by_genome:
                values = tuple(
                    float(value) for value in evaluate(dict(genome), seeds)
                )
                if len(values) != len(seeds):
                    raise ValueError("Evaluator must return one score per seed.")
                if any(not (-float("inf") < value < float("inf")) for value in values):
                    raise ValueError("Evaluator scores must be finite numbers.")
                scores_by_genome[key] = values
            return scores_by_genome[key]

        score(baseline_genome)
        population = [baseline_genome]
        while len(population) < population_size:
            population.append(self._random_genome())

        for _ in range(generations):
            ranked = sorted(
                population,
                key=lambda genome: self._mean(score(genome)),
                reverse=True,
            )
            elite_count = max(1, population_size // 3)
            elites = ranked[:elite_count]
            next_population = list(elites)
            seen = {self._key(genome) for genome in next_population}
            attempts = 0
            while len(next_population) < population_size:
                attempts += 1
                if attempts > population_size * 20:
                    child = self._random_genome()
                else:
                    parent = self._rng.choice(elites)
                    other = self._rng.choice(elites)
                    child = self._mutate(self._crossover(parent, other))
                key = self._key(child)
                if key in seen:
                    continue
                seen.add(key)
                next_population.append(child)
            population = next_population

        leaderboard = tuple(
            sorted(
                (
                    StrategyCandidateResult(dict(key), values)
                    for key, values in scores_by_genome.items()
                ),
                key=lambda candidate: self._mean(candidate.scores),
                reverse=True,
            )
        )
        best = max(
            (candidate.genome for candidate in leaderboard),
            key=lambda genome: self._mean(score(genome)),
        )
        return StrategyResult(
            baseline=baseline_genome,
            best=best,
            baseline_scores=score(baseline_genome),
            best_scores=score(best),
            leaderboard=leaderboard,
            evaluated=len(scores_by_genome),
        )

    def _validated(self, genome: Mapping[str, float]) -> Genome:
        if set(genome) != set(self.bounds):
            raise ValueError("Strategy parameters must match configured bounds.")
        validated = {name: float(value) for name, value in genome.items()}
        for name, value in validated.items():
            low, high = self.bounds[name]
            if not low <= value <= high:
                raise ValueError(f"Strategy parameter {name!r} is outside its bounds.")
            if name in self.integer_parameters and not value.is_integer():
                raise ValueError(f"Strategy parameter {name!r} must be an integer.")
        return validated

    def _random_genome(self) -> Genome:
        return {
            name: (
                float(self._rng.randint(int(low), int(high)))
                if name in self.integer_parameters
                else self._rng.uniform(low, high)
            )
            for name, (low, high) in self.bounds.items()
        }

    def _crossover(self, first: Genome, second: Genome) -> Genome:
        return {
            name: self._rng.choice((first[name], second[name]))
            for name in self.bounds
        }

    def _mutate(self, genome: Genome) -> Genome:
        child = dict(genome)
        name = self._rng.choice(tuple(self.bounds))
        low, high = self.bounds[name]
        if name in self.integer_parameters:
            choices = [
                value
                for value in range(int(low), int(high) + 1)
                if value != child[name]
            ]
            child[name] = float(self._rng.choice(choices))
        else:
            span = high - low
            child[name] = min(
                high,
                max(low, child[name] + self._rng.gauss(0, span * 0.2)),
            )
        return child

    @staticmethod
    def _mean(scores: Sequence[float]) -> float:
        return sum(scores) / len(scores)

    @staticmethod
    def _key(genome: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
        return tuple(sorted((name, round(value, 6)) for name, value in genome.items()))
