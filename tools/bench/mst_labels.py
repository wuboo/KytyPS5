#!/usr/bin/env python3
"""S446: GPU ms/flip per encoder label from a Metal System Trace export (metal-gpu-intervals as a list of rows).

usage: mst_labels.py <export.json|export.json.gz> <pid> --flips N [--window START_MS:END_MS] [--top 20] [--json]

Row layout (as exported by artifacts/449): [start_ns, duration_ns, channel, state, ?, ?, name, pid, command_buffer_id].
Rows of the "Command Buffer N:Compute Command" kind are the kernel-side view of the same work and are skipped.
A label written by the emulator (KYTY_GPU_LABELS) is found anywhere in the name: G.cs.<hash>.a<0|1>.<x>x<y>x<z>, G.rp.<W>x<H>.<n>.<id>,
I.<class>... , with an optional suffix .e=<cause>. Rows without one are class "unlabeled" (the MoltenVK name of the first command is kept).
Output: ms/flip per class, top dispatches/passes, gaps inside a command buffer per cause, channel overlap, pass->pass pairs with equal attachments.
"""
import argparse
import collections
import gzip
import json
import re
import sys

LABEL = re.compile(r'\b([GI]\.[A-Za-z]+[^\s|(]*)')
KERNEL_SIDE = re.compile(r'Command Buffer \d+:Compute Command')
MVK = re.compile(r'vkQueueSubmit MTLCommandBuffer on Queue \d+-\d+:vkCmd(\S+) (\S+)')
CAUSE = re.compile(r'\.e=([a-z]+)$')


def load(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        return json.load(f)


def parse_label(name):
    """-> (class_key, item_key, cause). class_key: G.cs.<hash> | G.rp.<WxH> | I.<name> | unlabeled:<MVK cmd>."""
    m = LABEL.search(name)
    if not m:
        v = MVK.match(name)
        return ('unlabeled:' + (v.group(1) if v else 'other')), name, None
    label = m.group(1)
    cause = None
    c = CAUSE.search(label)
    if c:
        cause = c.group(1)
        label = label[:c.start()]
    parts = label.split('.')
    if parts[0] == 'G' and parts[1] == 'cs' and len(parts) >= 3:
        key = '.'.join(parts[:3])                      # G.cs.<hash>
    elif parts[0] == 'G' and parts[1] == 'rp' and len(parts) >= 3:
        key = '.'.join(parts[:3])                      # G.rp.<WxH>
    else:
        key = '.'.join(parts[:2])                      # I.<class>, G.fill, G.clr
    return key, label, cause


def rows_of(data, pid, window):
    rows = []
    for r in data:
        if str(r[7]) != str(pid) or KERNEL_SIDE.match(r[6]):
            continue
        s, e = r[0] / 1e6, (r[0] + r[1]) / 1e6
        if s < window[0] or e > window[1]:
            continue
        key, label, cause = parse_label(r[6])
        rows.append({'s': s, 'e': e, 'ch': r[2], 'cb': r[8], 'key': key, 'label': label, 'cause': cause})
    rows.sort(key=lambda x: (x['s'], x['e']))
    return rows


def analyse(rows, flips, top=20):
    out = {'flips': flips}
    classes = collections.defaultdict(lambda: [0, 0.0])
    for x in rows:
        c = classes[x['key']]
        c[0] += 1
        c[1] += x['e'] - x['s']
    out['classes'] = {k: {'enc_per_flip': n / flips, 'ms_per_flip': d / flips}
                      for k, (n, d) in sorted(classes.items(), key=lambda kv: -kv[1][1])}
    out['total_ms_per_flip'] = sum(d for _, d in classes.values()) / flips
    items = sorted(rows, key=lambda x: -(x['e'] - x['s']))[:top]
    out['top'] = [{'label': x['label'], 'ch': x['ch'], 'ms': x['e'] - x['s']} for x in items]
    # Gaps: time not covered by any encoder between two rows of the same command buffer, per cause of the row that follows.
    gaps = collections.defaultdict(lambda: [0, 0.0])
    same = other = 0.0
    if rows:
        end, prev = rows[0]['e'], rows[0]
        for x in rows[1:]:
            if x['s'] > end:
                gap = x['s'] - end
                if x['cb'] == prev['cb']:
                    same += gap
                    g = gaps[x['cause'] or 'none']
                    g[0] += 1
                    g[1] += gap
                else:
                    other += gap
            if x['e'] > end:
                end, prev = x['e'], x
    out['gap_in_cb_ms_per_flip'] = same / flips
    out['gap_between_cb_ms_per_flip'] = other / flips
    out['gap_by_cause'] = {k: {'n_per_flip': n / flips, 'ms_per_flip': d / flips}
                           for k, (n, d) in sorted(gaps.items(), key=lambda kv: -kv[1][1])}
    # Channel overlap: busy time per channel versus the union of all of them.
    by_ch = collections.defaultdict(float)
    for x in rows:
        by_ch[x['ch']] += x['e'] - x['s']
    union = 0.0
    cur_s = cur_e = None
    for x in rows:
        if cur_e is None or x['s'] > cur_e:
            if cur_e is not None:
                union += cur_e - cur_s
            cur_s, cur_e = x['s'], x['e']
        else:
            cur_e = max(cur_e, x['e'])
    if cur_e is not None:
        union += cur_e - cur_s
    out['channels_ms_per_flip'] = {k: v / flips for k, v in sorted(by_ch.items())}
    out['union_ms_per_flip'] = union / flips
    out['overlap_ms_per_flip'] = (sum(by_ch.values()) - union) / flips
    # Pass -> pass pairs: two G.rp encoders in a row (nothing opened between them) with the same attachment key.
    pairs = collections.defaultdict(int)
    prev = None
    for x in rows:
        if x['key'].startswith('G.rp.') and x['ch'] in ('Vertex', 'Fragment'):
            if prev is not None and prev['key'] == x['key'] and prev['label'] != x['label']:
                pairs[x['key'] + ('  same CB' if prev['cb'] == x['cb'] else '  new CB')] += 1
            prev = x
        elif x['ch'] not in ('Fragment',):
            prev = None
    out['pass_pairs_per_flip'] = {k: v / flips for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('export')
    ap.add_argument('pid')
    ap.add_argument('--flips', type=int, required=True)
    ap.add_argument('--window', default='1000:9000')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args(argv)
    lo, hi = (float(v) for v in args.window.split(':'))
    result = analyse(rows_of(load(args.export), args.pid, (lo, hi)), args.flips, args.top)
    if args.json:
        json.dump(result, sys.stdout, indent=1)
        print()
        return 0
    print(f"{args.flips} flips, total {result['total_ms_per_flip']:.2f} ms/flip")
    for k, v in list(result['classes'].items())[:30]:
        print(f"  {k:36s} enc/flip {v['enc_per_flip']:7.1f}  ms/flip {v['ms_per_flip']:7.2f}")
    print(f"top {args.top}:")
    for t in result['top']:
        print(f"  {t['ms']:8.3f} ms  {t['ch']:9s} {t['label'][:80]}")
    print(f"gaps in CB {result['gap_in_cb_ms_per_flip']:.2f} ms/flip, between CB {result['gap_between_cb_ms_per_flip']:.2f} ms/flip")
    for k, v in result['gap_by_cause'].items():
        print(f"  e={k:6s} n/flip {v['n_per_flip']:6.1f} ms/flip {v['ms_per_flip']:.2f}")
    print(f"channels {result['channels_ms_per_flip']}, union {result['union_ms_per_flip']:.2f}, overlap {result['overlap_ms_per_flip']:.2f} ms/flip")
    for k, v in list(result['pass_pairs_per_flip'].items())[:10]:
        print(f"  pass->pass {k}: {v:.1f}/flip")
    return 0


if __name__ == '__main__':
    sys.exit(main())
