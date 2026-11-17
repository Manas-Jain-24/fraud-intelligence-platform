"""Label-independent uniform PaySim sample; preserves natural prevalence in expectation."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / ".cache/kagglehub/datasets/ealaxi/paysim1/versions/2/PS_20174392719_1491204439457_log.csv"


def prepare(source, output, rows=250000, seed=42):
    if rows < 100:
        raise ValueError("Use at least 100 rows")
    rng = np.random.default_rng(seed)
    reservoir = None
    source_rows = 0
    digest = sha256()
    # Hash the exact input without loading a second copy into memory.
    with Path(source).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    for chunk in pd.read_csv(source, chunksize=250000):
        source_rows += len(chunk)
        chunk = chunk.drop(columns=["isFlaggedFraud"], errors="ignore")
        # Every source row, including every fraud, has the same inclusion probability.
        chunk["_priority"] = rng.random(len(chunk))
        candidates = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
        reservoir = candidates.nsmallest(min(rows, len(candidates)), "_priority")
    frame = reservoir.drop(columns="_priority").sort_values("step", kind="stable").reset_index(drop=True)
    frame["errorBalanceOrig"] = frame.newbalanceOrig + frame.amount - frame.oldbalanceOrg
    frame["errorBalanceDest"] = frame.oldbalanceDest + frame.amount - frame.newbalanceDest
    frame["balanceDrainRatio"] = frame.amount / (frame.oldbalanceOrg + 1.)
    frame["isTransfer"] = (frame.type == "TRANSFER").astype(int)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    manifest = {"source_file": Path(source).name, "source_sha256": digest.hexdigest(),
                "source_rows": source_rows, "sample_rows": len(frame), "frauds": int(frame.isFraud.sum()),
                "prevalence": float(frame.isFraud.mean()), "seed": seed,
                "sampling": "uniform random priority sample, independent of label; all transaction types",
                "history": "Sample is unsuitable for reconstructing complete account histories."}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "data/paysim_natural.csv")
    parser.add_argument("--rows", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.input, args.output, args.rows, args.seed)
