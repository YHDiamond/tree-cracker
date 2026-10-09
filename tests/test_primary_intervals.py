"""Verify the primary kernel's integer intervals against independent LCG filtering.

These tests exercise arithmetic and batch/lane coverage without a CUDA runtime.
The complete CUDA program must additionally match baseline candidate sets on GPU.
"""

import random
import unittest


A = 0x5DEECE66D
B = 11
M = 1 << 48
H = 1 << 44
MASK = M - 1


def interval_offset(k, boundary, first_z):
    quotient, remainder = divmod(M, A)
    split_quotient, split_remainder = divmod(remainder << 16, A)
    whole = k * quotient + (k >> 16) * split_quotient
    residual = ((k >> 16) * split_remainder + (k & 0xFFFF) * remainder
                + boundary - first_z)
    assert 0 <= whole < (1 << 64)
    assert -(1 << 63) <= residual + A - 1 < (1 << 63)
    # C++ integer division truncates toward zero for a negative numerator.
    rounded = ((residual + A - 1) // A if residual >= 0
               else -((-residual) // A))
    return whole + rounded


def block_count(count):
    high = (count >> 24) * A
    low = ((high & 0xFFFFFF) << 24) + (count & 0xFFFFFF) * A
    assert high < (1 << 64) and low < (1 << 64)
    return (high >> 24) + (low >> 48) + 2


def interval_end(begin, z, first_z):
    excess = ((first_z + begin * A) & MASK) - z * H
    assert 0 <= excess < A
    return begin + H // A + int(H % A > excess)


def enumerated_offsets(first_seed, z, count, grid=None, lanes=256):
    first_z = (first_seed * A + B) & MASK
    grid = block_count(count) if grid is None else grid
    values = []
    for block in range(grid):
        k = block
        while k < block_count(count):
            begin = interval_offset(k, z * H, first_z)
            if begin >= count:
                break
            end = interval_end(begin, z, first_z)
            begin, end = max(begin, 0), min(max(end, 0), count)
            for lane in range(lanes):
                values.extend(range(begin + lane, end, lanes))
            k += grid
    return values


class PrimaryIntervalTests(unittest.TestCase):
    def assert_matches_brute_force(self, first_seed, z, count, **kwargs):
        actual = enumerated_offsets(first_seed, z, count, **kwargs)
        expected = [offset for offset in range(count)
                    if (((first_seed + offset) * A + B) & MASK) >> 44 == z]
        self.assertEqual(len(actual), len(set(actual)), "duplicate offset")
        self.assertEqual(sorted(actual), expected)

    def test_all_coordinate_nibbles_and_batch_edges(self):
        for x in range(16):
            for z in range(16):
                for start, count in ((0, 1), (697, 699), (H - 65536, 65536)):
                    self.assert_matches_brute_force(x * H + start, z, count)

    def test_exact_z_boundaries_and_negative_ceil(self):
        inverse = pow(A, -1, M)
        for z in range(16):
            for boundary in (z * H, (z + 1) * H):
                for delta in (-1, 0, 1):
                    seed = (((boundary + delta) & MASK) - B) * inverse & MASK
                    self.assert_matches_brute_force(seed, z, min(4096, H - (seed & (H - 1))))

    def test_grid_stride_and_lane_partition(self):
        for grid in (1, 2, 3):
            for lanes in (1, 32, 256, 1024):
                for z in (0, 7, 15):
                    self.assert_matches_brute_force(15 * H + 123456, z, 65536,
                                                   grid=grid, lanes=lanes)

    def test_full_domain_endpoint_arithmetic(self):
        rng = random.Random(0)
        max_wrap = ((M - 1) * A + M - 1) // M
        cases = [0, 1, 65535, 65536, max_wrap, max_wrap + (1 << 31)]
        cases += [rng.randrange(max_wrap + 1) for _ in range(10000)]
        for k in cases:
            for boundary, first_z in ((0, MASK), (M, 0), (7 * H, 11 * H)):
                # Independent arbitrary-precision division is the oracle.
                expected = -(-(k * M + boundary - first_z) // A)
                self.assertEqual(interval_offset(k, boundary, first_z), expected)
            for z, first_z in ((0, MASK), (15, 0), (7, 11 * H)):
                begin = interval_offset(k, z * H, first_z)
                expected = -(-(k * M + (z + 1) * H - first_z) // A)
                self.assertEqual(interval_end(begin, z, first_z), expected)

    def test_launch_bounds_for_full_and_small_batches(self):
        rng = random.Random(1)
        counts = [1, 17, 256, 697, 699, (1 << 32) - 1, 1 << 32, H, M]
        counts += [rng.randrange(1, M + 1) for _ in range(10000)]
        for count in counts:
            self.assertEqual(block_count(count), count * A // M + 2)
            for first_z in (0, MASK):
                last_wrap = (first_z + (count - 1) * A) // M
                self.assertLess(last_wrap, block_count(count))


if __name__ == "__main__":
    unittest.main()
