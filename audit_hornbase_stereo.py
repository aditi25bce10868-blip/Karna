"""Audit original HornBase stereo WAV files without modifying them.

Run from the live-audio-classifier repository root:
  python audit_hornbase_stereo.py --manifest emergency_datset/full_dataset_manifest.csv

If the manifest is elsewhere, pass its path with --manifest.
The script prefers source_path (original audio), not prepared_path (which may
already have been converted to mono). It reports missing paths rather than
silently substituting prepared files.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import soundfile as sf


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-15))


def db(x):
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def resolve_original(source_path, manifest_path, repo_root):
    p = Path(str(source_path))
    candidates = [p] if p.is_absolute() else [
        repo_root / p,
        manifest_path.parent / p,
        manifest_path.parent.parent / p,
        repo_root / "emergency_datset" / p,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def analyse(path):
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    n, channels = audio.shape
    out = {
        "actual_channels": channels,
        "actual_sample_rate": sr,
        "duration_seconds": n / sr if sr else np.nan,
    }
    if channels < 2:
        out.update({
            "left_rms_db": np.nan, "right_rms_db": np.nan,
            "right_minus_left_db": np.nan, "stereo_correlation": np.nan,
            "mid_rms_db": np.nan, "side_rms_db": np.nan,
            "side_to_mid_db": np.nan, "mono_cancellation_db": np.nan,
        })
        return out

    # First two channels; retain any extra channel count in actual_channels.
    left = audio[:, 0].astype(np.float64)
    right = audio[:, 1].astype(np.float64)
    l_rms, r_rms = rms(left), rms(right)
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    corr = np.nan if np.std(left) < 1e-10 or np.std(right) < 1e-10 else float(np.corrcoef(left, right)[0, 1])
    mid_rms, side_rms = rms(mid), rms(side)
    out.update({
        "left_rms_db": db(l_rms),
        "right_rms_db": db(r_rms),
        "right_minus_left_db": db(r_rms) - db(l_rms),
        "stereo_correlation": corr,
        "mid_rms_db": db(mid_rms),
        "side_rms_db": db(side_rms),
        "side_to_mid_db": db(side_rms / max(mid_rms, 1e-12)),
        "mono_cancellation_db": db(mid_rms / max((l_rms + r_rms) / 2, 1e-12)),
    })
    return out


def main():
    ap = argparse.ArgumentParser(description="Audit original HornBase stereo recordings")
    ap.add_argument("--manifest", default="emergency_datset/full_dataset_manifest.csv")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out-dir", default="emergency_datset/dataset_preparation/stereo_audit")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    required = {"source_type", "class_name", "source_path"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_cols)}\nFound: {list(df.columns)}")

    horn = df[df["source_type"].astype(str).str.strip().str.lower().eq("hornbase")].copy()
    print(f"HornBase rows in manifest: {len(horn)}")
    results, failures = [], []
    for i, (_, row) in enumerate(horn.iterrows(), 1):
        path = resolve_original(row["source_path"], manifest_path, repo_root)
        if path is None:
            failures.append({"class_name": row["class_name"], "source_path": row["source_path"], "reason": "original source_path not found"})
            continue
        try:
            rec = row.to_dict()
            rec["resolved_original_path"] = str(path)
            rec.update(analyse(path))
            results.append(rec)
        except Exception as exc:
            failures.append({"class_name": row["class_name"], "source_path": row["source_path"], "reason": repr(exc)})
        if i % 50 == 0:
            print(f"Checked {i}/{len(horn)} manifest rows...")

    result = pd.DataFrame(results)
    fail_df = pd.DataFrame(failures)
    result.to_csv(out_dir / "stereo_channel_audit.csv", index=False)
    fail_df.to_csv(out_dir / "missing_or_failed_originals.csv", index=False)

    if result.empty:
        print("No original audio files were successfully analysed.")
        print("Check missing_or_failed_originals.csv and the source_path values.")
        return

    metrics = ["stereo_correlation", "left_rms_db", "right_rms_db", "right_minus_left_db", "side_to_mid_db", "mono_cancellation_db"]
    result.groupby("class_name")[metrics].agg(["count", "mean", "median", "std"]).to_csv(out_dir / "stereo_class_summary.csv")

    # Group by useful metadata columns if they exist in the manifest.
    group_cols = [c for c in ["class_name", "position", "pattern", "device", "scenario_group_id"] if c in result.columns]
    if group_cols:
        result.groupby(group_cols)[metrics].agg(["count", "mean", "median", "std"]).to_csv(out_dir / "stereo_metadata_summary.csv")

    print("\n========== ORIGINAL HORNBASE STEREO AUDIT ==========")
    print(f"Successfully analysed: {len(result)}")
    print(f"Missing/failed originals: {len(fail_df)}")
    print("\nChannel counts:")
    print(result["actual_channels"].value_counts().sort_index().to_string())
    print("\nSample rates:")
    print(result["actual_sample_rate"].value_counts().sort_index().to_string())
    print("\nPer-class mean stereo correlation:")
    print(result.groupby("class_name")["stereo_correlation"].mean().round(4).to_string())
    print("\nPer-class mean right-minus-left level (dB):")
    print(result.groupby("class_name")["right_minus_left_db"].mean().round(3).to_string())
    print("\nPer-class mean side-to-mid level (dB):")
    print(result.groupby("class_name")["side_to_mid_db"].mean().round(3).to_string())
    print("\nOutput folder:", out_dir)
    print("Files: stereo_channel_audit.csv, stereo_class_summary.csv, stereo_metadata_summary.csv (if metadata available), missing_or_failed_originals.csv")

if __name__ == "__main__":
    main()
