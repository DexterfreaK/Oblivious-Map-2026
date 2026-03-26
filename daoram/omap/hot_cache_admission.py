"""Hot-cache admission layers for OMAP hot-node caches.

Only the admission sampler itself is modeled here. If the supplied utility
function has global sensitivity at most ``utility_sensitivity`` under the
caller-defined neighboring relation, then the sampling step in
``ExponentialMechanismHotCacheAdmissionLayer`` matches the standard
exponential-mechanism form:

    Pr[out] proportional to exp((epsilon * utility) / (2 * utility_sensitivity))

This module does not claim an end-to-end privacy guarantee for the surrounding
cache, traversal, or ORAM protocol. Those layers may still leak information
through access patterns, timing, cache state, or any later non-oblivious
processing unless they are analyzed separately.
"""

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple


@dataclass(frozen=True)
class HotCacheAdmissionCandidate:
    """Small snapshot of a node used by the admission layer."""

    key: Any
    metadata: Dict[str, Any]

    @property
    def access_count(self) -> int:
        return int(self.metadata.get("access_count", 0))

    @property
    def secret_user_id(self) -> Any:
        return self.metadata.get("secret_user_id")


@dataclass(frozen=True)
class HotCacheAdmissionDecision:
    """Result produced by an admission layer."""

    admit: bool
    evict_key: Any = None


class HotCacheAdmissionLayer:
    """Abstract admission layer between a hot cache and ORAM."""

    def decide(
        self,
        *,
        candidate: HotCacheAdmissionCandidate,
        residents: Sequence[HotCacheAdmissionCandidate],
        capacity: int,
    ) -> HotCacheAdmissionDecision:
        raise NotImplementedError


def secret_user_id_access_distance(
    candidate: HotCacheAdmissionCandidate,
    resident: HotCacheAdmissionCandidate,
) -> float:
    """
    Default distance for the exponential mechanism.

    Cross-secret candidates tie with the arriving node. Within the same
    secret bucket, residents with higher access counts are increasingly
    protected.
    """
    if candidate.secret_user_id != resident.secret_user_id:
        return 0.0

    candidate_access = max(1, candidate.access_count)
    resident_access = max(0, resident.access_count)
    if resident_access <= candidate_access:
        return 0.0

    return min(1.0, float(resident_access - candidate_access) / float(max(resident_access, 1)))


def secret_user_id_access_utility(
    candidate: HotCacheAdmissionCandidate,
    resident: HotCacheAdmissionCandidate,
) -> float:
    """
    Default utility for the exponential mechanism.

    Cross-secret residents tie with rejecting the arriving node. Within the
    same secret bucket, lower-access residents become more evictable than the
    arriving node, while higher-access residents are protected.
    """
    if candidate.secret_user_id != resident.secret_user_id:
        return 0.0

    candidate_access = max(1, candidate.access_count)
    resident_access = max(1, resident.access_count)
    scale = max(candidate_access, resident_access)
    return max(-1.0, min(1.0, float(candidate_access - resident_access) / float(scale)))


class ScoreBasedHotCacheAdmissionLayer(HotCacheAdmissionLayer):
    """Stable deterministic admission layer based on access count and secret id."""

    def _numeric_secret_user_id(self, candidate: HotCacheAdmissionCandidate) -> float:
        secret_user_id = candidate.secret_user_id
        return float(secret_user_id) if isinstance(secret_user_id, (int, float)) else 0.0

    def _score(self, candidate: HotCacheAdmissionCandidate) -> Tuple[int, float]:
        return candidate.access_count, self._numeric_secret_user_id(candidate)

    def decide(
        self,
        *,
        candidate: HotCacheAdmissionCandidate,
        residents: Sequence[HotCacheAdmissionCandidate],
        capacity: int,
    ) -> HotCacheAdmissionDecision:
        if capacity <= 0:
            return HotCacheAdmissionDecision(admit=False)

        if len(residents) < capacity:
            return HotCacheAdmissionDecision(admit=True)

        victim_index = None
        victim = None
        victim_score = None
        for index, resident in enumerate(residents):
            score = (*self._score(resident), index)
            if victim_score is None or score < victim_score:
                victim_index = index
                victim = resident
                victim_score = score

        if victim is None or victim_index is None:
            return HotCacheAdmissionDecision(admit=False)

        if self._score(candidate) > self._score(victim):
            return HotCacheAdmissionDecision(admit=True, evict_key=victim.key)

        return HotCacheAdmissionDecision(admit=False)


class RejectAllHotCacheAdmissionLayer(HotCacheAdmissionLayer):
    """Admission layer that disables all cache promotions."""

    def decide(
        self,
        *,
        candidate: HotCacheAdmissionCandidate,
        residents: Sequence[HotCacheAdmissionCandidate],
        capacity: int,
    ) -> HotCacheAdmissionDecision:
        return HotCacheAdmissionDecision(admit=False)


class ExponentialMechanismHotCacheAdmissionLayer(HotCacheAdmissionLayer):
    """
    Probabilistic cache admission using the normalized exponential mechanism.

    Guarantee boundary:
    if ``utility_fn`` has global sensitivity at most ``utility_sensitivity``
    under the caller's neighboring relation, then this sampler is
    ``epsilon``-DP over the candidate-set output distribution. That statement
    applies only to this sampling step, not to the wider cache or ORAM system.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        utility_sensitivity: float = 1.0,
        utility_fn: Callable[[HotCacheAdmissionCandidate, HotCacheAdmissionCandidate], float] = None,
        distance_fn: Callable[[HotCacheAdmissionCandidate, HotCacheAdmissionCandidate], float] = None,
        rng: random.Random = None,
    ):
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        if utility_sensitivity <= 0:
            raise ValueError("utility_sensitivity must be positive.")
        if utility_fn is not None and distance_fn is not None:
            raise ValueError("Provide at most one of utility_fn or distance_fn.")
        self._epsilon = float(epsilon)
        self._utility_sensitivity = float(utility_sensitivity)
        if utility_fn is not None:
            self._utility_fn = utility_fn
        elif distance_fn is not None:
            self._utility_fn = lambda candidate, resident: -float(distance_fn(candidate, resident))
        else:
            self._utility_fn = secret_user_id_access_utility
        self._rng = rng or random.Random()

    def _utility(
        self,
        candidate: HotCacheAdmissionCandidate,
        other: HotCacheAdmissionCandidate,
    ) -> float:
        utility = 0.0 if candidate.key == other.key else float(self._utility_fn(candidate, other))
        if utility < -1.0 or utility > 1.0:
            raise ValueError("utility_fn must return a value in [-1, 1].")
        return utility

    def _weight(
        self,
        candidate: HotCacheAdmissionCandidate,
        other: HotCacheAdmissionCandidate,
    ) -> float:
        scaled_utility = (self._epsilon * self._utility(candidate, other)) / (2.0 * self._utility_sensitivity)
        return math.exp(scaled_utility)

    def decide(
        self,
        *,
        candidate: HotCacheAdmissionCandidate,
        residents: Sequence[HotCacheAdmissionCandidate],
        capacity: int,
    ) -> HotCacheAdmissionDecision:
        if capacity <= 0:
            return HotCacheAdmissionDecision(admit=False)

        if len(residents) < capacity:
            return HotCacheAdmissionDecision(admit=True)

        candidate_set = [candidate, *residents]
        weights = [self._weight(candidate, other) for other in candidate_set]
        total_weight = sum(weights)
        if total_weight <= 0:
            return HotCacheAdmissionDecision(admit=False)

        cutoff = self._rng.random() * total_weight
        cumulative = 0.0
        victim = candidate
        for other, weight in zip(candidate_set, weights):
            cumulative += weight
            if cutoff <= cumulative:
                victim = other
                break

        if victim.key == candidate.key:
            return HotCacheAdmissionDecision(admit=False)

        return HotCacheAdmissionDecision(admit=True, evict_key=victim.key)


def make_hot_cache_candidate(key: Any, metadata: Dict[str, Any]) -> HotCacheAdmissionCandidate:
    """Create an immutable admission snapshot from mutable node metadata."""
    return HotCacheAdmissionCandidate(key=key, metadata=copy.deepcopy(metadata))
