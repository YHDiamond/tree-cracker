"""CPU parity checks for the original and indexed-coordinate filter2/3 scans.

Run with ``python tests/test_filter_scan_parity.py``. These are CPU ports of
the Java 1.16.1 Forest scan/predicates, not a CUDA build or end-to-end cracker.
Only the five committed public/development fixtures are read; no world saves,
blind data or inference seed catalogue are used. Validation seeds are used
only to construct positive test windows, never to limit a cracking search.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
import struct
import unittest


MULTIPLIER = 0x5DEECE66D
ADDEND = 11
MASK = (1 << 48) - 1
MAX_CALLS = 11 * 21  # biomeMaxRandomCalls(Forest, v1_16_1).
FLOAT_POINT_TWO = struct.unpack("f", struct.pack("f", 0.2))[0]
FLOAT_POINT_ONE = struct.unpack("f", struct.pack("f", 0.1))[0]
FIXTURES = json.loads((Path(__file__).resolve().parents[1]
                      / "Test Data/positions_types_fixture_candidates.json").read_text())["fixtures"]


class JavaRandom:
    """src/RNG Logic.cuh semantics, including Java bounded-int rejection."""
    def __init__(self, internal_state):
        self.seed = internal_state & MASK

    def copy(self):
        return JavaRandom(self.seed)

    def skip(self, calls):
        multiplier, addend = 1, 0
        power_multiplier, power_addend = MULTIPLIER, ADDEND
        # A full-period 48-bit LCG makes negative skips modulo 2**48 valid.
        calls &= MASK
        while calls:
            if calls & 1:
                multiplier = multiplier * power_multiplier & MASK
                addend = (addend * power_multiplier + power_addend) & MASK
            power_addend = (power_addend * (power_multiplier + 1)) & MASK
            power_multiplier = power_multiplier * power_multiplier & MASK
            calls >>= 1
        self.seed = (self.seed * multiplier + addend) & MASK
        return self

    def next(self, bits):
        self.seed = (self.seed * MULTIPLIER + ADDEND) & MASK
        return self.seed >> (48 - bits)

    def next_int(self, bound):
        value = self.next(31)
        if bound & (bound - 1) == 0:
            return bound * value >> 31
        while True:
            remainder = value % bound
            if value - remainder + bound - 1 < 1 << 31:
                return remainder
            value = self.next(31)

    def next_float(self):
        return self.next(24) / (1 << 24)

    def next_long(self):
        first, second = self.next(32), self.next(32)
        if second & (1 << 31):
            second -= 1 << 32
        return ((first << 32) + second) & ((1 << 64) - 1)


def next_tree_type(rng):
    # getNextTreeType uses nextFloat, not nextInt(5/10), in 1.16.1.
    if rng.next_float() < FLOAT_POINT_TWO:
        return "Birch"
    if rng.next_float() < FLOAT_POINT_ONE:
        return "Fancy_Oak"
    return "Oak"


@dataclass(frozen=True)
class Tree:
    x: int
    z: int
    types: tuple[str, ...]
    height: int | None = None
    # Indices refer to SetOfLeafStates' 16-call RNG mask, not Y/X/Z order.
    leaves: tuple[tuple[int, int], ...] = ()


def trees_for(fixture):
    return tuple(Tree(t["x"] & 15, t["z"] & 15, (t["type"],))
                 for t in fixture["input"]["trees"])


def matches_after_coordinates(rng, tree, attributes):
    if not tree.types:
        return True
    kind = next_tree_type(rng)
    if kind not in tree.types:
        return False
    if not attributes:
        return True
    if kind == "Fancy_Oak":
        height = 3 + rng.next_int(12) + rng.next_int(1)
        return tree.height is None or tree.height == height
    height = (5 if kind == "Birch" else 4) + rng.next_int(3) + rng.next_int(1)
    if tree.height is not None and tree.height != height:
        return False
    rng.skip(2)  # SetOfLeafStates' v1_16_1 advancement.
    for index, wanted in tree.leaves:
        leaf_rng = rng.copy().skip(index)
        if leaf_rng.next_int(2) != wanted:
            return False
    rng.skip(16)
    return True


def original_scan(seed, trees, attributes=False, radius=MAX_CALLS):
    """Original filter2/3: private RNG per tree, full inclusive scan."""
    rng = JavaRandom(seed).skip(-radius)
    found = 0
    for _ in range(-radius, radius + 1):
        for index, tree in enumerate(trees):
            tree_rng = rng.copy()
            if (tree_rng.next_int(16) == tree.x
                    and tree_rng.next_int(16) == tree.z
                    and matches_after_coordinates(tree_rng, tree, attributes)):
                found |= 1 << index
        rng.skip(1)
    return found


def optimized_scan(seed, trees, attributes=False, radius=MAX_CALLS):
    """Direct X/Z lookup with the same inclusive scan and RNG advancement."""
    lookup = {(tree.x, tree.z): index for index, tree in enumerate(trees)}
    assert len(lookup) == len(trees)  # TreeChunk merges types at one position.
    rng = JavaRandom(seed).skip(-radius)
    found = 0
    all_found = (1 << len(trees)) - 1
    for _ in range(-radius, radius + 1):
        if found == all_found:
            break
        coordinate_rng = rng.copy()
        x = coordinate_rng.next_int(16)
        rng = coordinate_rng.copy()
        z = coordinate_rng.next_int(16)
        index = lookup.get((x, z))
        if index is None or found & (1 << index):
            continue
        if matches_after_coordinates(coordinate_rng.copy(), trees[index], attributes):
            found |= 1 << index
    return found


def coordinate_type_state(tree, choices):
    """Construct one predicate's internal state, not a world-seed search."""
    while True:
        post_x = (tree.x << 44) | choices.getrandbits(44)
        rng = JavaRandom(post_x)
        if rng.next_int(16) == tree.z and next_tree_type(rng) in tree.types:
            return JavaRandom(post_x).skip(-1).seed


def feature_state(fixture):
    """Validation-only upstream population/feature RNG initialization."""
    world_seed = int(fixture["validation_only"]["world_seed"])
    first = fixture["input"]["trees"][0]
    x, z = (first["x"] // 16) * 16, (first["z"] // 16) * 16
    rng = JavaRandom(world_seed ^ MULTIPLIER)
    a, b = rng.next_long() | 1, rng.next_long() | 1
    population_seed = (x * a + z * b) ^ world_seed
    return JavaRandom((population_seed + 80001) ^ MULTIPLIER).seed


def constrain_to_state(tree, state):
    """Add exact unit-test attributes generated at an existing valid match."""
    rng = JavaRandom(state)
    rng.next_int(16)
    rng.next_int(16)
    kind = next_tree_type(rng)
    assert kind in tree.types
    if kind == "Fancy_Oak":
        return replace(tree, height=3 + rng.next_int(12) + rng.next_int(1))
    height = (5 if kind == "Birch" else 4) + rng.next_int(3) + rng.next_int(1)
    rng.skip(2)
    generated = tuple(rng.next_int(2) for _ in range(16))
    return replace(tree, height=height, leaves=tuple(enumerate(generated[4:], 4)))


class ScanParity(unittest.TestCase):
    coverage = Counter()

    def compare(self, label, seed, trees, attributes=False, radius=MAX_CALLS):
        expected = original_scan(seed, trees, attributes, radius)
        actual = optimized_scan(seed, trees, attributes, radius)
        self.assertEqual(expected, actual, (label, hex(seed), radius, attributes))
        self.coverage["mask_comparisons"] += 1
        self.coverage["filter3" if attributes else "filter2"] += 1
        if expected:
            self.coverage["nonempty_masks"] += 1
        else:
            self.coverage["empty_masks"] += 1
        if expected == (1 << len(trees)) - 1:
            self.coverage["accepted_masks"] += 1
        return expected

    def test_java_lcg_golden_values_and_inverse(self):
        # java.util.Random(0).nextLong(), a known Java RNG reference value.
        value = JavaRandom(MULTIPLIER).next_long()
        self.assertEqual(value - (1 << 64), -4962768465676381896)
        self.assertEqual(JavaRandom(MULTIPLIER).next_int(16), 11)
        for state in (0, 1, MASK, 1 << 44, (1 << 44) - 1):
            for calls in (0, 1, 2, MAX_CALLS, 2 * MAX_CALLS + 1):
                self.assertEqual(JavaRandom(state).skip(calls).skip(-calls).seed, state)

    def test_five_fixture_random_and_boundary_states(self):
        choices = random.Random(0x16_01_48)
        for fixture in FIXTURES:
            trees = trees_for(fixture)
            states = (0, 1, MASK, 1 << 44, (1 << 44) - 1, 1 << 47)
            states += tuple(choices.getrandbits(48) for _ in range(32))
            for state in states:
                for attributes in (False, True):
                    self.compare(fixture["id"], state, trees, attributes)

    def test_constructed_matches_at_inclusive_window_boundaries(self):
        choices = random.Random(0xC001D00D)
        for fixture in FIXTURES:
            trees = trees_for(fixture)
            for index, tree in enumerate(trees):
                state = coordinate_type_state(tree, choices)
                constrained = constrain_to_state(tree, state)
                for delta in (-232, -231, -230, 0, 230, 231, 232):
                    center = JavaRandom(state).skip(delta).seed
                    for attributes in (False, True):
                        found = self.compare(fixture["id"], center, trees, attributes)
                        if abs(delta) <= MAX_CALLS:
                            self.assertTrue(found & (1 << index))
                            self.coverage["inclusive_boundary_or_interior_matches"] += 1
                # A radius-zero singleton is a fully accepted scan, and exact
                # height/leaf constraints ensure attributes are actually tested.
                for attributes in (False, True):
                    self.assertEqual(self.compare(fixture["id"], state, (constrained,), attributes, 0), 1)
                wrong_height = replace(constrained, height=constrained.height + 1)
                self.assertEqual(self.compare(fixture["id"], state, (wrong_height,), True, 0), 0)
                if constrained.leaves:
                    index_leaf, bit = constrained.leaves[0]
                    wrong_leaf = replace(constrained, leaves=((index_leaf, 1 - bit),))
                    self.assertEqual(self.compare(fixture["id"], state, (wrong_leaf,), True, 0), 0)

    def test_validation_only_real_fixture_positive_windows(self):
        for fixture in FIXTURES:
            trees = trees_for(fixture)
            center = JavaRandom(feature_state(fixture)).skip(MAX_CALLS).seed
            for attributes in (False, True):
                mask = self.compare(fixture["id"], center, trees, attributes)
                self.assertEqual(mask, (1 << len(trees)) - 1, fixture["id"])
                self.coverage["accepted_full_fixture_masks"] += 1

    def test_unknown_types_alternatives_and_tree_order(self):
        choices = random.Random(0xA17E)
        for fixture in FIXTURES:
            trees = trees_for(fixture)
            variants = (tuple(reversed(trees)),
                        tuple(replace(t, types=()) for t in trees),
                        tuple(replace(t, types=("Oak", "Fancy_Oak", "Birch")) for t in trees))
            state = coordinate_type_state(trees[0], choices)
            for variant in variants:
                for attributes in (False, True):
                    self.compare(fixture["id"], state, variant, attributes)
            for attributes in (False, True):
                self.assertEqual(self.compare(fixture["id"], state, (), attributes), 0)

    @classmethod
    def tearDownClass(cls):
        print("\nParity coverage:", json.dumps(dict(cls.coverage), sort_keys=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
