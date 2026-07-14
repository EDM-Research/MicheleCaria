from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
from ultralytics import YOLO


def run_experiment(run_idx: int, base_output_dir: Path) -> dict:
    DATA_YAML = Path("175_replacement_crack") / "dataset.yaml"
    MODEL_NAME = "yolov8n.pt"
    IMG_SIZE = 512
    BATCH_SIZE = 8
    EPOCHS = 20
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    PATIENCE = 10
    SEED = 1000 + run_idx

    run_output_dir = base_output_dir / f"run_{run_idx}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_output_dir / "log.txt"
    summary_path = run_output_dir / "summary.json"

    def log(msg: str = "") -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if log_path.exists():
        log_path.unlink()

    log(f"=== RUN {run_idx} (seed={SEED}) ===")

    model = YOLO(MODEL_NAME)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        patience=PATIENCE,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-4,
        weight_decay=1e-4,
        project=str(base_output_dir),
        name=f"run_{run_idx}",
        exist_ok=True,
        val=True,
        plots=False,
        verbose=True,
        seed=SEED,
        deterministic=False,
    )

    best_weights = run_output_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_weights))

    val_metrics = best_model.val(data=str(DATA_YAML), split="val", imgsz=IMG_SIZE, batch=BATCH_SIZE, device=DEVICE)
    test_metrics = best_model.val(data=str(DATA_YAML), split="test", imgsz=IMG_SIZE, batch=BATCH_SIZE, device=DEVICE)

    summary = {
        "run_idx": run_idx,
        "seed": SEED,
        "val": {
            "map50": float(val_metrics.box.map50),
            "map50_95": float(val_metrics.box.map),
        },
        "test": {
            "map50": float(test_metrics.box.map50),
            "map50_95": float(test_metrics.box.map),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(f"val  mAP50={summary['val']['map50']:.4f}  mAP50-95={summary['val']['map50_95']:.4f}")
    log(f"test mAP50={summary['test']['map50']:.4f}  mAP50-95={summary['test']['map50_95']:.4f}")

    return summary


def main() -> None:
    base_output_dir = Path("model_results") / "detection" / "175_replacement_crack"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_log = base_output_dir / "aggregate_summary.txt"
    if aggregate_log.exists():
        aggregate_log.unlink()

    def log_agg(msg: str = "") -> None:
        print(msg)
        with aggregate_log.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    summaries: list[dict] = []
    for run_idx in range(1, 11):
        summaries.append(run_experiment(run_idx, base_output_dir))

    val_map50 = [s["val"]["map50"] for s in summaries]
    val_map = [s["val"]["map50_95"] for s in summaries]
    test_map50 = [s["test"]["map50"] for s in summaries]
    test_map = [s["test"]["map50_95"] for s in summaries]

    log_agg(f"Runs: {len(summaries)}")
    log_agg(f"val  mAP50    mean={statistics.mean(val_map50):.4f}  std={statistics.stdev(val_map50):.4f}")
    log_agg(f"val  mAP50-95 mean={statistics.mean(val_map):.4f}  std={statistics.stdev(val_map):.4f}")
    log_agg(f"test mAP50    mean={statistics.mean(test_map50):.4f}  std={statistics.stdev(test_map50):.4f}")
    log_agg(f"test mAP50-95 mean={statistics.mean(test_map):.4f}  std={statistics.stdev(test_map):.4f}")

    (base_output_dir / "aggregate_summary.json").write_text(
        json.dumps({
            "n_runs": len(summaries),
            "val_map50": {"mean": statistics.mean(val_map50), "std": statistics.stdev(val_map50)},
            "val_map50_95": {"mean": statistics.mean(val_map), "std": statistics.stdev(val_map)},
            "test_map50": {"mean": statistics.mean(test_map50), "std": statistics.stdev(test_map50)},
            "test_map50_95": {"mean": statistics.mean(test_map), "std": statistics.stdev(test_map)},
            "runs": summaries,
        }, indent=2),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()