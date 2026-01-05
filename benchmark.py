import os
import json
import time
import datetime
import pandas as pd

from src.signature_detect.loader import Loader
from src.signature_detect.judger import Judger
from src.signature_detect.cropper import Cropper
from src.signature_detect.extractor import Extractor


def run_pipeline(file_path, loader, extractor, cropper, judger):
    start = time.time()
    try:
        masks = loader.get_masks(file_path)
        is_signed = False

        for mask in masks:
            labeled_mask = extractor.extract(mask)
            results = cropper.run(labeled_mask)

            for result in results.values():
                if judger.judge(result["cropped_mask"]):
                    is_signed = True
                    break
            if is_signed:
                break

        elapsed = time.time() - start
        return {"file": file_path, "is_signed": is_signed, "time": elapsed}
    except Exception as e:
        return {"file": file_path, "is_signed": None, "error": str(e)}


def benchmark(signed_dir, unsigned_dir):
    loader = Loader()
    extractor = Extractor(amplfier=15)
    cropper = Cropper(min_region_size=5000, border_ratio=0.02)
    judger = Judger(size_ratio=[1, 10], pixel_ratio=[0.1, 50])

    hp = {
        "Loader": {
            "low_threshold": loader.low_threshold,
            "high_threshold": loader.high_threshold,
        },
        "Extractor": {
            "outlier_weight": extractor.outlier_weight,
            "outlier_bias": extractor.outlier_bias,
            "amplfier": extractor.amplfier,
            "min_area_size": extractor.min_area_size,
        },
        "Cropper": {
            "min_region_size": cropper.min_region_size,
            "border_ratio": cropper.border_ratio,
        },
        "Judger": {
            "size_ratio": judger.size_ratio,
            "pixel_ratio": judger.pixel_ratio,
        },
    }

    # Collect files
    signed_files = [
        os.path.join(signed_dir, f)
        for f in os.listdir(signed_dir)
        if os.path.isfile(os.path.join(signed_dir, f))
    ]
    unsigned_files = [
        os.path.join(unsigned_dir, f)
        for f in os.listdir(unsigned_dir)
        if os.path.isfile(os.path.join(unsigned_dir, f))
    ]
    all_files = [(f, True) for f in signed_files] + [(f, False) for f in unsigned_files]

    total_files = len(all_files)
    results = []

    print(f"\nStarting benchmark on {total_files} files...")
    print("---------------------------------------------------")

    for idx, (fpath, expected) in enumerate(all_files, start=1):
        print(f"[{idx}/{total_files}] Processing: {os.path.basename(fpath)}")
        res = run_pipeline(fpath, loader, extractor, cropper, judger)
        res["expected"] = expected
        results.append(res)

    # Compute metrics
    tp = sum(1 for r in results if r["is_signed"] and r["expected"])
    tn = sum(1 for r in results if not r["is_signed"] and not r["expected"])
    fp = sum(1 for r in results if r["is_signed"] and not r["expected"])
    fn = sum(1 for r in results if not r["is_signed"] and r["expected"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    summary = {
        "total": len(results),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    output = {
        "hyperparameters": hp,
        "results": results,
        "summary": summary,
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "outputs/benchmark"
    os.makedirs(out_dir, exist_ok=True)
    out_json_path = os.path.join(out_dir, f"benchmark_{timestamp}.json")
    out_csv_path = os.path.join(out_dir, f"benchmark_{timestamp}.xlsx")

    # Save CSV
    df = pd.DataFrame(results)
    df.to_excel(out_csv_path, index=False)

    with open(out_json_path, "w") as f:
        json.dump(output, f, indent=4)

    print("Benchmark completed.")
    print("---------------------------------------------------")
    print(json.dumps(summary, indent=4))
    print(f"Results saved to: {out_json_path}\n")


if __name__ == "__main__":
    signed_dir = "Datasets/signed/images_test"
    unsigned_dir = "Datasets/unsigned/images"
    benchmark(signed_dir, unsigned_dir)
