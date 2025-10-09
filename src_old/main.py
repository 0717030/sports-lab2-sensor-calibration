# src/main.py
import argparse
from pathlib import Path
from src.pipeline import PipelineConfig, run

def run_single(csv_path: Path, out_root: Path, action: str, args):
    cfg = PipelineConfig(
        csv=str(csv_path),
        out=str(out_root / action),
        action=action,
        alpha=args.alpha,
        gyro_thresh_dps=args.gyro_thresh_dps,
        min_stationary_sec=args.min_stationary_sec,
        use_mag=args.use_mag,
        export_calibrated_csv=(not args.no_export)
    )
    print(f"[RUN] action={action:<9} csv={csv_path} -> out={cfg.out}")
    run(cfg)

def infer_action_from_name(name: str) -> str:
    n = name.lower()
    if "pendulum" in n:
        return "pendulum"
    if "elevator" in n or "lift" in n:
        return "elevator"
    if "square" in n:
        return "square"
    return "custom"

def main():
    ap = argparse.ArgumentParser("lab2")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--csv", help="single CSV path (e.g., data/xxx.csv)")
    g.add_argument("--data-dir", default="data", help="batch mode: read all CSV in this dir (default: data)")

    ap.add_argument("--action", choices=["pendulum","elevator","square","custom"],
                    help="when using --csv, force the action; otherwise auto-infer by filename")
    ap.add_argument("--out-root", default="results", help="output root folder (default: results)")

    # calibration / filter
    ap.add_argument("--alpha", type=float, default=0.02, help="complementary filter accel/mag blend (0~1)")
    ap.add_argument("--gyro_thresh_dps", type=float, default=3.0, help="stationary detection threshold [deg/s]")
    ap.add_argument("--min_stationary_sec", type=float, default=0.5, help="min stationary duration [s]")
    ap.add_argument("--use_mag", action="store_true", help="use magnetometer for yaw correction (optional)")
    ap.add_argument("--no-export", action="store_true",default=True, help="do NOT export calibrated CSVs")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.csv:
        csv_path = Path(args.csv)
        action = args.action or infer_action_from_name(csv_path.name)
        run_single(csv_path, out_root, action, args)
    else:
        data_dir = Path(args.data_dir)
        csvs = sorted(p for p in data_dir.glob("*.csv"))
        if not csvs:
            raise SystemExit(f"No CSV found in {data_dir.resolve()}")
        for csv_path in csvs:
            action = infer_action_from_name(csv_path.name)
            run_single(csv_path, out_root, action, args)

if __name__ == "__main__":
    main()