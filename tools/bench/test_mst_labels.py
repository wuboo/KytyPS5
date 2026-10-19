#!/usr/bin/env python3
"""S446: mst_labels.py on synthetic exports (3 cases) and on the real export of artifacts/449 when it is there.
run: python3 -I tools/bench/test_mst_labels.py   (MUTANT=1 -> the label regex is disabled: the tests must fail)"""
import gzip
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mst_labels as M  # noqa: E402

if os.environ.get('MUTANT') == '1':
    M.LABEL = __import__('re').compile(r'\bNEVERMATCH(x)')

PID = '4242'


def row(start_ms, dur_ms, ch, name, cb):
    return [int(start_ms * 1e6), int(dur_ms * 1e6), ch, 'Active', 0, 0, name, PID, cb]


def mvk(cmd, extra=''):
    return f'vkQueueSubmit MTLCommandBuffer on Queue 0-0:vkCmd{cmd} ComputeEncoder      ( kyty_emulator ({PID}) ){extra}'


class SyntheticTests(unittest.TestCase):
    def test_classes_sum_per_label(self):
        data = [row(1000, 4, 'Compute', 'G.cs.999c5a2d.a1.120x135x1 ' + mvk('Dispatch'), 'a'),
                row(1010, 1, 'Compute', 'G.cs.999c5a2d.a1.120x135x1', 'a'),
                row(1020, 0.5, 'Compute', 'I.up.64kb.e=gb', 'a'),
                row(1030, 2, 'Vertex', mvk('BeginRendering'), 'a'),                  # unlabeled
                [0, 5_000_000, 'Compute', 'Active', 0, 0, 'Command Buffer 1:Compute Command 0', PID, 'a']]  # kernel side: skipped
        r = M.analyse(M.rows_of(data, PID, (0, 9000)), flips=2)
        self.assertAlmostEqual(r['classes']['G.cs.999c5a2d']['ms_per_flip'], 2.5)
        self.assertAlmostEqual(r['classes']['I.up']['ms_per_flip'], 0.25)
        self.assertAlmostEqual(r['classes']['unlabeled:BeginRendering']['ms_per_flip'], 1.0)
        self.assertAlmostEqual(r['total_ms_per_flip'], 3.75)
        self.assertEqual(r['classes']['G.cs.999c5a2d']['enc_per_flip'], 1.0)

    def test_gap_by_cause_and_cb(self):
        data = [row(1000, 1, 'Compute', 'G.cs.11111111.a0.1x1x1', 'a'),
                row(1002, 1, 'Compute', 'G.cs.22222222.a0.1x1x1.e=cs', 'a'),     # 1 ms gap in the same CB, cause cs
                row(1010, 1, 'Compute', 'G.cs.33333333.a0.1x1x1.e=sub', 'b')]    # gap between command buffers
        r = M.analyse(M.rows_of(data, PID, (0, 9000)), flips=1)
        self.assertAlmostEqual(r['gap_in_cb_ms_per_flip'], 1.0)
        self.assertAlmostEqual(r['gap_between_cb_ms_per_flip'], 7.0)
        self.assertAlmostEqual(r['gap_by_cause']['cs']['ms_per_flip'], 1.0)
        self.assertNotIn('sub', r['gap_by_cause'])

    def test_overlap_and_pass_pairs(self):
        data = [row(1000, 4, 'Compute', 'G.cs.999c5a2d.a1.120x135x1', 'a'),
                row(1001, 2, 'Vertex', 'G.rp.1920x1080.4.0', 'a'),                # overlaps the dispatch
                row(1003, 1, 'Vertex', 'G.rp.1920x1080.4.1', 'a'),                # pass -> pass, same attachments
                row(1020, 1, 'Vertex', 'G.rp.960x540.1.2', 'b')]
        r = M.analyse(M.rows_of(data, PID, (0, 9000)), flips=1)
        self.assertAlmostEqual(r['overlap_ms_per_flip'], 3.0)
        self.assertEqual(r['pass_pairs_per_flip'], {'G.rp.1920x1080  same CB': 1.0})
        self.assertEqual([t['label'] for t in r['top']][0], 'G.cs.999c5a2d.a1.120x135x1')


class RealExportTests(unittest.TestCase):
    PATH = os.environ.get('MST449', '/Users/bart/PycharmProjects/experiments/ps5/kbench-results/rt-loop/artifacts/449/mst-iv-K1.json.gz')

    def test_unlabeled_sum_equals_enc449(self):
        if not os.path.exists(self.PATH):
            self.skipTest('artifacts/449 not available')
        data = M.load(self.PATH)
        r = M.analyse(M.rows_of(data, '66880', (1000, 9000)), flips=186)   # K1: flips 6377..6563
        self.assertTrue(all(k.startswith('unlabeled') for k in r['classes']), list(r['classes'])[:5])
        self.assertGreater(r['classes']['unlabeled:Dispatch']['ms_per_flip'], 10)
        # enc449.py: the same rows, summed per MoltenVK type
        total = sum(v['ms_per_flip'] for v in r['classes'].values())
        self.assertAlmostEqual(total, r['total_ms_per_flip'])


if __name__ == '__main__':
    unittest.main()
