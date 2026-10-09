"""Compare shared-prefix traversal to exhaustive valid/invalid mask enumeration.

The synthetic transition has variable, type-dependent RNG skip lengths. Matching
returns at most one position, as required by the production input builder's
coordinate merging. GPU checks additionally validate the actual tree predicates.
"""

import random
import unittest


A = 0x5DEECE66D
B = 11
MASK = (1 << 48) - 1


def step(state, valid):
    # Different skip lengths exercise branches analogous to ordinary/fancy trees.
    calls = (3 + (state >> 17) % 3) if not valid else (7 + (state >> 19) % 14)
    for _ in range(calls):
        state = (state * A + B) & MASK
    return state


def match(state, positions):
    site = (state >> 28) % (positions + 2)
    # The extra condition represents a tree-specific type/height/leaf predicate.
    return (1 << site) if site < positions and state % 7 != 0 else 0


def exhaustive_masks(state, attempts, positions, max_attempts=11):
    all_found = (1 << positions) - 1
    for mask in range(1 << max_attempts):
        if mask.bit_count() < positions or mask >> attempts:
            continue
        current, found = state, 0
        for level in range(attempts):
            valid = bool(mask & (1 << level))
            if valid:
                found |= match(current, positions)
            current = step(current, valid)
        if found == all_found:
            return True
    return False


def shared_prefixes(state, attempts, positions):
    all_found = (1 << positions) - 1
    pending = []
    found, level = 0, 0
    while True:
        if found == all_found:
            return True
        remaining = attempts - level
        if remaining and found.bit_count() + remaining >= positions:
            valid_found = found | match(state, positions)
            if valid_found == all_found:
                return True
            visit_valid = valid_found.bit_count() + remaining - 1 >= positions
            visit_invalid = found.bit_count() + remaining - 1 >= positions
            if visit_invalid:
                invalid_state = step(state, False)
                if not visit_valid:
                    state = invalid_state
                    level += 1
                    continue
                pending.append((invalid_state, found, level + 1))
                assert len(pending) <= attempts
            if visit_valid:
                state = step(state, True)
                found = valid_found
                level += 1
                continue
        if not pending:
            return False
        state, found, level = pending.pop()


class TreechunkPrefixTests(unittest.TestCase):
    def test_random_variable_skip_models_match_every_mask(self):
        rng = random.Random(2)
        passes = failures = 0
        for attempts in (1, 4, 7, 10, 11):
            for positions in (1, 2, 4, 5, 11):
                for _ in range(20):
                    state = rng.randrange(MASK + 1)
                    expected = exhaustive_masks(state, attempts, positions)
                    self.assertEqual(shared_prefixes(state, attempts, positions), expected)
                    passes += expected
                    failures += not expected
        self.assertGreater(passes, 0)
        self.assertGreater(failures, 0)

    def test_more_observations_than_attempts_cannot_pass(self):
        for positions in range(2, 12):
            self.assertFalse(shared_prefixes(123456, positions - 1, positions))

    def test_empty_observation_set_and_empty_suffix(self):
        self.assertTrue(shared_prefixes(123456, 0, 0))
        self.assertFalse(shared_prefixes(123456, 0, 1))


if __name__ == "__main__":
    unittest.main()
