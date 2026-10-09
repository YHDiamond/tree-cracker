"""Check complete logical-worker coverage for treechunk CUDA grid indexing."""

import unittest


def decode(index, calls_per_seed, masks_per_seed):
    calls = index % calls_per_seed
    index //= calls_per_seed
    mask = index % masks_per_seed
    return index // masks_per_seed, mask, calls


class TreechunkIndexTests(unittest.TestCase):
    def test_pilot_candidates_beyond_32bit_boundary(self):
        calls, masks, candidates = 232, 2048, 34099
        total = candidates * calls * masks
        self.assertGreater(total, 1 << 32)
        for index in (0, (1 << 32) - 1, 1 << 32, total - 1):
            seed, mask, call = decode(index, calls, masks)
            self.assertEqual((seed * masks + mask) * calls + call, index)
            self.assertLess(seed, candidates)
        self.assertEqual(decode(total - 1, calls, masks),
                         (candidates - 1, masks - 1, calls - 1))
        self.assertNotEqual(decode(total - 1, calls, masks),
                            decode((total - 1) & 0xFFFFFFFF, calls, masks))

    def test_grid_stride_covers_small_domain_exactly_once(self):
        for total in (1, 255, 256, 257, 65536):
            for stride in (1, 17, 256, 1024):
                values = [i for lane in range(stride)
                          for i in range(lane, total, stride)]
                self.assertEqual(sorted(values), list(range(total)))

    def test_grid_limit_does_not_truncate_large_domain(self):
        stride = ((1 << 31) - 1) * 256
        total = 134217727 * 232 * 2048
        self.assertGreater(total, stride)
        for index in (0, stride - 1, stride, total - 1):
            lane, iteration = index % stride, index // stride
            self.assertLess(lane, stride)
            self.assertEqual(lane + iteration * stride, index)


if __name__ == "__main__":
    unittest.main()
