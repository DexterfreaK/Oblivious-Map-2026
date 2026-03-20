import random

from daoram.omap import (
    ExponentialMechanismHotCacheAdmissionLayer,
    HotCacheAdmissionCandidate,
    secret_user_id_access_utility,
)


def _candidate(key, access_count, secret_user_id):
    return HotCacheAdmissionCandidate(
        key=key,
        metadata={
            "access_count": access_count,
            "secret_user_id": secret_user_id,
        },
    )


class TestHotCacheAdmissionLayer:
    def test_default_utility_prefers_lower_access_same_secret_resident(self):
        candidate = _candidate("x", access_count=30, secret_user_id="A")
        lower = _candidate("low", access_count=10, secret_user_id="A")
        higher = _candidate("high", access_count=90, secret_user_id="A")
        cross_secret = _candidate("cross", access_count=1, secret_user_id="B")

        assert secret_user_id_access_utility(candidate, lower) > 0.0
        assert secret_user_id_access_utility(candidate, higher) < 0.0
        assert secret_user_id_access_utility(candidate, cross_secret) == 0.0

    def test_default_exponential_mechanism_biases_toward_lower_access_eviction(self):
        layer = ExponentialMechanismHotCacheAdmissionLayer(epsilon=2.0, rng=random.Random(0))
        candidate = _candidate("x", access_count=30, secret_user_id="A")
        lower = _candidate("low", access_count=10, secret_user_id="A")
        higher = _candidate("high", access_count=90, secret_user_id="A")
        cross_secret = _candidate("cross", access_count=1, secret_user_id="B")

        counts = {"x": 0, "low": 0, "high": 0, "cross": 0}
        for _ in range(5000):
            decision = layer.decide(
                candidate=candidate,
                residents=[lower, higher, cross_secret],
                capacity=3,
            )
            victim_key = "x" if not decision.admit else decision.evict_key
            counts[victim_key] += 1

        assert counts["low"] > counts["x"]
        assert counts["x"] > counts["high"]
        assert abs(counts["cross"] - counts["x"]) < 250

    def test_distance_compatibility_mode_is_still_supported(self):
        layer = ExponentialMechanismHotCacheAdmissionLayer(
            epsilon=1.0,
            distance_fn=lambda candidate, resident: 0.0,
            rng=random.Random(1),
        )
        candidate = _candidate("x", access_count=30, secret_user_id="A")
        resident = _candidate("resident", access_count=10, secret_user_id="A")

        decision = layer.decide(candidate=candidate, residents=[resident], capacity=1)
        assert decision.admit is False
