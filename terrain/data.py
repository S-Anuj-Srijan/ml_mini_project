"""BorealTC preparation: recording-wise splits and non-overlapping windows."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["ASPHALT", "FLOORING", "ICE", "SANDY_LOAM", "SNOW"]
CHANNELS = ["wx", "wy", "wz", "ax", "ay", "az", "curL", "curR", "velL", "velR"]
WINDOW = 170

def extract_features(x):
    """Six statistics per channel -> 60 features; no terrain label is an input."""
    if x.ndim != 3 or x.shape[1:] != (10, WINDOW) or not np.isfinite(x).all():
        raise ValueError("Expected finite sensor windows with shape (N, 10, 170)")
    return np.concatenate([x.mean(2), x.std(2), x.min(2), x.max(2),
                           np.sqrt(np.mean(x*x,2)),
                           np.sqrt(np.mean(np.diff(x,axis=2)**2,2))],axis=1).astype(np.float32)

def read_recording(imu_path):
    imu = pd.read_csv(imu_path)
    pro = pd.read_csv(imu_path.with_name(imu_path.name.replace("imu_", "pro_")))
    for df, cols in [(imu, ["time"]+CHANNELS[:6]), (pro, ["time"]+CHANNELS[6:])]:
        if not set(cols).issubset(df.columns) or not np.isfinite(df[cols].to_numpy()).all():
            raise ValueError(f"Invalid sensor columns or values in {imu_path}")
        if not (np.diff(df.time.to_numpy()) > 0).all():
            raise ValueError(f"Timestamps must be strictly increasing: {imu_path}")
    # The IMU is supplied at 100 Hz. Restrict to common coverage, then causal hold.
    imu = imu[(imu.time >= pro.time.iloc[0]) & (imu.time <= pro.time.iloc[-1])]
    if not np.allclose(np.diff(imu.time), .01, atol=1e-5):
        raise ValueError("Expected the dataset's 100 Hz IMU grid")
    fused = pd.merge_asof(imu, pro, on="time", direction="backward")
    if fused[CHANNELS].isna().any().any():
        raise ValueError("Alignment produced missing measurements")
    return fused

def split_runs(paths, seed=42):
    """Choose holdout recordings separately within each class, before windowing."""
    rng = np.random.default_rng(seed)
    mapping = {}
    for label in CLASSES:
        runs = sorted(p for p in paths if p.parent.name == label)
        if len(runs) < 5:
            raise ValueError(f"Need at least five recordings for {label}")
        order = rng.permutation(len(runs))
        n_holdout = max(1, round(.2*len(runs)))
        for i, idx in enumerate(order):
            mapping[str(runs[idx])] = "test" if i < n_holdout else ("val" if i < 2*n_holdout else "train")
    return mapping

def prepare(seed=42):
    raw = ROOT/"data/raw/borealtc"
    paths = sorted(p for c in CLASSES for p in (raw/c).glob("imu_*.csv"))
    mapping = split_runs(paths, seed)
    xs, ys, groups, splits, starts = [], [], [], [], []
    records = []
    for p in paths:
        df = read_recording(p)
        values = df[CHANNELS].to_numpy(dtype=np.float32)
        group = p.parent.name + "/" + p.stem.removeprefix("imu_")
        count = 0
        for start in range(0, len(values)-WINDOW+1, WINDOW):
            xs.append(values[start:start+WINDOW].T)
            ys.append(CLASSES.index(p.parent.name)); groups.append(group)
            splits.append(mapping[str(p)]); starts.append(float(df.time.iloc[start]))
            count += 1
        if not count:
            raise ValueError(f"Recording too short: {p}")
        records.append(dict(run=group, split=mapping[str(p)], rows=len(df), windows=count))
    X = np.array(xs,dtype=np.float32)
    data = dict(X=X, features=extract_features(X), y=np.array(ys,dtype=np.int64),
                groups=np.array(groups), split=np.array(splits), start_seconds=np.array(starts))
    for s in ["train","val","test"]:
        if set(data["y"][data["split"]==s]) != set(range(5)):
            raise ValueError(f"Missing class in {s}")
    output = ROOT/"data/processed"
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/"windows.npz",**data)
    summary = dict(dataset="BorealTC", seed=seed, sample_rate_hz=100, window_samples=WINDOW,
                   channels=CHANNELS, classes=CLASSES, windows=len(X), recordings=len(paths),
                   class_counts={s:{c:int(np.sum((data["split"]==s)&(data["y"]==i))) for i,c in enumerate(CLASSES)} for s in ["train","val","test"]},
                   records=records)
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!="records"},indent=2))
    return data

def load_data():
    with np.load(ROOT/"data/processed/windows.npz",allow_pickle=False) as d:
        return {k:d[k] for k in d.files}

if __name__ == "__main__":
    prepare()
