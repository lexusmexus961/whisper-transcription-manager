"""How well does the gallery recognize? - with the file itself as a blind spot.

The obvious evaluation is worthless: If you ask "which person matches this window?"
while the windows of the same file lie in the gallery, the window will find itself
in doubt. A dream number would result that says nothing about new recordings.

Therefore **Leave-one-out over the FILE**, not over the vector: For each recording,
the entire file is removed from the gallery and then asked who speaks in it. This is
exactly the situation with a freshly transcribed file - the person is known, but this
recording is not.

Three numbers are reported that answer different questions:

**Hit rate** - how many recordings have the correct person above the threshold?
This is the question "who was there".

**Accuracy** - how often is the BEST person also the correct one? This measures
confusions and is independent of the threshold.

**False alarms** - how often is a WRONG person above the threshold? This is
the most expensive type of error: a wrong name in a recording that you never
listen to again.

Caveat: The persons are usually named on groups that the model itself
formed. This selection embellishes the numbers - especially the **false alarms**
and confusions between two named persons are reliable, not the absolute hit rate.

    ./.venv/bin/python tools/speaker_quality.py
    ./.venv/bin/python tools/speaker_quality.py --threshold 0.5
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def nrm(A):
    """The ONE normalization - see whisper_tm/core/speaker.py.

    There were thirteen copies of it, some float64 with clamping, some
    float32 additive, some with axis=1 instead of axis=-1. The difference
    lies in the third decimal place - exactly where thresholds 0.55 / 0.62 / 0.75
    decide.
    """
    from whisper_tm.core.speaker import nrm as _nrm_core

    return _nrm_core(A)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument(
        "--single",
        action="store_true",
        help="against individual voices instead of against the center of mass",
    )
    p.add_argument(
        "--share",
        type=float,
        default=None,
        help="Rule \"at least this SHARE of a person's voices "
        'are above the threshold" instead of "the best '
        'value is above it". More robust against the effect that '
        "a person with 300 voices has 300 attempts and one "
        "with three only three.",
    )
    args = p.parse_args()

    from whisper_tm.config import config

    config.load()
    from whisper_tm.core import speaker as speaker_core
    from whisper_tm.database.speaker import from_blob, speaker_db

    threshold = (
        args.threshold if args.threshold is not None else speaker_core.THRESHOLD_ACROSS_FILES
    )
    db = speaker_db()
    db.initialize()

    # Load everything once — vector, person, file.
    with db.connection() as c:
        lines = [
            dict(r)
            for r in c.execute(
                "SELECT v.id, v.abs_path, v.v_separated, v.v_raw, p.name "
                "FROM speaker_vectors v "
                "JOIN speaker_assignment sa ON sa.vector_id = v.id "
                "JOIN speaker_people  p ON p.id = sa.person_id "
                "WHERE v.model = ?",
                (speaker_core.MODEL_BRAND,),
            ).fetchall()
        ]
    if not lines:
        raise SystemExit("No named voices in storage.")

    X, who, file = [], [], []
    for r in lines:
        v = from_blob(r["v_separated"])
        if v is None:
            v = from_blob(r["v_raw"])
        if v is None or not v.size:
            continue
        X.append(v)
        who.append(r["name"])
        file.append(r["abs_path"])
    X = nrm(np.stack(X).astype(np.float32))
    who = np.array(who)
    file = np.array(file)
    people = sorted(set(who))

    print("\n=== Leave-one-out over the file ===")
    print(
        f"{len(X)} named windows · {len(set(file))} files · "
        f"{len(people)} persons · threshold {threshold:.2f} · "
        f"{'individual voices' if args.single else 'center of mass'}\n"
    )
    for n, k in Counter(who).most_common():
        print(f"  {n:<14} {k:4d} windows from {len(set(file[who == n])):3d} files")

    hits = false_alarm = checked = confused = 0
    without_gallery = 0
    per_person: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [found, checked]
    confusion: Counter = Counter()

    for path in sorted(set(file)):
        inside = file == path
        true_val = who[inside][0]
        outside = ~inside
        # If nothing remains of the sought person, the question cannot
        # be asked - this is not a defeat, but a gap in the material.
        if not (who[outside] == true_val).any():
            without_gallery += 1
            continue

        if args.share is not None:
            # Not "one value is enough", but "a share of the person must
            # agree". Otherwise the best value rewards the person with the
            # most voices: one person has 300 attempts, another only three.
            S = np.zeros(len(people), dtype=np.float32)
            for i, n in enumerate(people):
                G = X[outside][who[outside] == n]
                if not len(G):
                    continue
                # best share over all windows of this file
                A = (X[inside] @ G.T) >= threshold  # (windows, voices)
                S[i] = float(A.mean(axis=1).max())
            # The comparison below runs against `threshold`; here the size
            # is a SHARE. So bring it to the same scale.
            S = np.where(args.share <= S, 1.0, 0.0)
            comparison = 1.0
        elif args.single:
            # Best individual value per person
            S = np.array(
                [float((X[inside] @ X[outside][who[outside] == n].T).max()) for n in people]
            )
            comparison = threshold
        else:
            Z = nrm(np.stack([nrm(X[outside][who[outside] == n]).mean(0) for n in people]))
            S = (X[inside] @ Z.T).max(0)  # best window per person
            comparison = threshold

        checked += 1
        j = int(np.argmax(S))
        best = people[j]
        own = people.index(true_val)

        per_person[true_val][1] += 1
        if S[own] >= comparison:
            hits += 1
            per_person[true_val][0] += 1
        if best != true_val:
            confused += 1
            confusion[(true_val, best)] += 1
        # False alarm: a WRONG person above the threshold
        wrong = [people[i] for i in range(len(people)) if i != own and S[i] >= comparison]
        if wrong:
            false_alarm += 1

    print(f"\n  {'checked files':<34} {checked:5d}")
    if without_gallery:
        print(f"  {'skipped without comparison voice':<34} {without_gallery:5d}")
    print(
        f"  {'correct person above threshold':<34} {hits:5d}  "
        f"= {100 * hits / max(checked, 1):5.1f} %   ← hit rate"
    )
    print(
        f"  {'best person was the correct one':<34} "
        f"{checked - confused:5d}  = {100 * (checked - confused) / max(checked, 1):5.1f} %"
        f"   ← accuracy"
    )
    print(
        f"  {'with wrong person above threshold':<34} {false_alarm:5d}  "
        f"= {100 * false_alarm / max(checked, 1):5.1f} %   ← false alarms"
    )

    print(f"\n  {'Person':<14} {'found':>10} {'of':>6}")
    for n in sorted(per_person, key=lambda x: -per_person[x][1]):
        g, total = per_person[n]
        print(f"  {n:<14} {g:10d} {total:6d}   {100 * g / max(total, 1):5.0f} %")

    if confusion:
        print("\n  Confusions (true → guessed):")
        for (a, b), k in confusion.most_common(12):
            print(f"    {a:<14} → {b:<14} {k:3d}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
