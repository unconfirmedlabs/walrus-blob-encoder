#!/usr/bin/env python3
# Copyright (c) Unconfirmed Labs, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runs a blob-ID command several times and reports its best run as one JSON line.

usage: measure.py [--repeat N] [--expect BLOB_ID] [--label TEXT] -- COMMAND...

Each run goes through /usr/bin/time; the first 43-character word of each output line that looks
like a blob ID (or the word after "blob_id") is checked against --expect on every run. The
best run (lowest wall time) is reported with its CPU time (user + system), the maximum
resident set size over all runs, and the last JSON object the command printed, if any.
"""

import json
import re
import subprocess
import sys

ID = re.compile(r"^[A-Za-z0-9_-]{43}$")


def main() -> int:
    args = sys.argv[1:]
    repeat, expect, label = 3, None, None
    while args and args[0] != "--":
        flag, value, args = args[0], args[1], args[2:]
        if flag == "--repeat":
            repeat = int(value)
        elif flag == "--expect":
            expect = value
        elif flag == "--label":
            label = value
        else:
            print(__doc__, file=sys.stderr)
            return 2
    command = args[1:]
    runs = []
    for _ in range(repeat):
        timed = ["/usr/bin/time", "-f", "TIME %e %U %S %M"] + command
        proc = subprocess.run(timed, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout, proc.stderr, file=sys.stderr)
            return 1
        wall, user, system, rss = proc.stderr.strip().splitlines()[-1].split()[1:]
        ids, stats = [], None
        for line in proc.stdout.splitlines():
            words = line.split()
            if line.startswith("{"):
                stats = json.loads(line)
            elif words and words[0] == "blob_id" and len(words) > 1:
                ids.append(words[1])
            elif words and ID.match(words[0]):
                ids.append(words[0])
        if expect is not None and ids != [expect]:
            print(f"blob ID mismatch: {ids} != {expect}", file=sys.stderr)
            return 1
        runs.append((float(wall), float(user) + float(system), int(rss), ids, stats))
    best = min(runs, key=lambda run: run[0])
    print(json.dumps({
        "label": label,
        "wall_s": best[0],
        "cpu_s": round(best[1], 2),
        "max_rss_kib": max(run[2] for run in runs),
        "walls": [run[0] for run in runs],
        "blob_id": best[3][0] if best[3] else None,
        "id_checked": expect is not None,
        "stats": best[4],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
