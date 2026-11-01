import threading
import unittest
from pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_overlap_and_order(self):
        decoding = threading.Event()
        second_sample = threading.Event()
        def sample(i):
            if i == 1:
                self.assertTrue(decoding.wait(2))
                second_sample.set()
            return i
        def decode(i):
            if i == 0:
                decoding.set()
                self.assertTrue(second_sample.wait(2))
            return i * 2
        self.assertEqual(run_pipeline(range(3), sample, decode, max_inflight=2), [0, 2, 4])

    def test_failure_propagates_and_stops(self):
        seen = []
        def decode(i):
            if i == 0: raise ValueError('decode failed')
            return i
        def sample(i):
            seen.append(i)
            return i
        with self.assertRaisesRegex(ValueError, 'decode failed'):
            run_pipeline(range(4), sample, decode, max_inflight=1)
        self.assertEqual(seen, [0])

    def test_cancel_prevents_start(self):
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            run_pipeline([1], lambda x: x, lambda x: x, cancelled=lambda: True)

    def test_invalid_bound(self):
        with self.assertRaises(ValueError):
            run_pipeline([], lambda x: x, lambda x: x, max_inflight=0)


if __name__ == '__main__':
    unittest.main()
