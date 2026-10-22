"""Copy verified, small report exhibits into a new shareable results directory."""
import argparse
import json
from pathlib import Path
import shutil

from final_eval.analysis.audit import BASE, MODEL, PREPARED, check
from final_eval.inference.client import public_config
from final_eval.utils.artifacts import atomic_json, claim_output, file_hash, source_fingerprint


def publish(audit_dir, output_dir):
    audit = Path(audit_dir)
    report = json.loads((audit / "verification.json").read_text(encoding="utf-8"))
    check(report["passed"] and report["retokenized"] and len(report["weights"]) == 5,
          "A complete --retokenize --checkpoints audit is required")
    check(report["core_source_fingerprint"] == source_fingerprint(), "Core source changed since verification")
    for name, info in {**report["evidence"], **report["weights"]}.items():
        check(file_hash(name) == info["sha256"], f"Evidence changed since audit: {name}")
    target = Path(output_dir); claim_output(target)
    copies = {}
    def add(source, destination):
        source = Path(source)
        check(source.is_file(), f"Missing exhibit: {source}")
        copies[destination] = source
    for name in ("accuracy", "paired", "coverage", "macro_accuracy", "grading_coverage", "manifest"):
        suffix = ".json" if name == "manifest" else ".csv"
        add(Path("outputs/final_eval/two_reader_strict_v1") / (name + suffix), "tables/full_cohort/" + name + suffix)
    for reader, directory in [("gptoss", BASE / "reports_gptoss_v1"),
                              ("qwen", Path("outputs/final_eval/main_qwen8192_v1/reports_qwen_completecase_v1"))]:
        for name in ("accuracy.csv", "coverage.csv", "paired.csv", "manifest.json"):
            add(directory / name, f"tables/common_success/{reader}/{name}")
    for name in ("compression_timing.csv", "macro_accuracy.csv", "grading_coverage.csv", "retention_prediction.csv",
                 "retention_bins.csv", "target_usage.csv", "claims_ledger.csv"):
        add(BASE / "analysis_gptoss_v1" / name, "tables/gptoss_diagnostics/" + name)
    for name in ("pairwise.csv", "coverage.csv", "prevalence.csv"):
        add(Path("outputs/offline_preparation_20261021/agreement") / name, "tables/label_agreement/" + name)
    add(audit / "compute_estimates.csv", "tables/compute_estimates.csv")
    add(audit / "verification.json", "verification.json")
    add("outputs/final_eval/two_reader_strict_v1/two_reader_accuracy.png", "figures/two_reader_accuracy.png")
    add("outputs/final_eval/feature_baseline/scorer_roc.png", "figures/scorer_roc.png")
    add("outputs/offline_preparation_20261021/agreement/agreement.png", "figures/label_agreement.png")
    for name in ("accuracy_tokens_0.png", "retention_accuracy_0.png", "retention_accuracy_1.png"):
        add(BASE / "analysis_gptoss_v1" / name, "figures/gptoss_" + name)
    metadata = {
        "preparation.json": PREPARED / "manifest.json",
        "regrading.json": Path("outputs/offline_preparation_20261021/regrading_summary.json"),
        "gsmic_verification.json": Path("outputs/offline_preparation_20261021/gsmic_verification.json"),
        "training.json": MODEL / "training_metadata.json",
        "scorer_report.json": MODEL / "scorer_report.json",
        "feature_report.json": Path("outputs/final_eval/feature_baseline/feature_report.json"),
        "main_cohort.json": BASE / "cohort.csv.cohort.json",
        "compression_check.json": BASE / "compression_check.json",
        "kaggle_environment.json": BASE / "environment.json",
        "gptoss_main_check.json": BASE / "target_gptoss/check.json",
        "qwen_main_check.json": Path("outputs/final_eval/main_qwen8192_v1/target_qwen/check.json"),
        "gptoss_response_manifest.json": BASE / "target_gptoss/responses.csv.manifest.json",
        "qwen_response_manifest.json": Path("outputs/final_eval/main_qwen8192_v1/target_qwen/responses.csv.manifest.json"),
    }
    hf_verification = Path("outputs/hf_release/remote_release_verification.json")
    if hf_verification.is_file():
        check(json.loads(hf_verification.read_text(encoding="utf-8"))["complete"], "HF release verification is incomplete")
        metadata["huggingface_release.json"] = hf_verification
    for destination, source in metadata.items():
        if destination.endswith("response_manifest.json"):
            public_config(json.loads(source.read_text(encoding="utf-8"))["config"])
        add(source, "metadata/" + destination)
    for path in sorted((MODEL / "profiles").glob("*.json")):
        add(path, "profiles/training/" + path.name)
    for directory in sorted(BASE.glob("*.profiles")):
        for path in sorted(directory.glob("*.json")):
            add(path, "profiles/compression/" + directory.name + "/" + path.name)
    for name in ("last_run_attempt.json", "last_setup_attempt.json", "restore_constraints.txt"):
        add(BASE / name, "profiles/" + name)
    # Resolve every source and validate every checksum before creating the publication.
    manifest = {name: {"source": path.as_posix(), "sha256": file_hash(path), "bytes": path.stat().st_size}
                for name, path in sorted(copies.items())}
    target.mkdir(parents=True)
    for name, path in copies.items():
        destination = target / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        check(file_hash(destination) == manifest[name]["sha256"], f"Copy verification failed: {name}")
    atomic_json(target / "manifest.json", {"files": manifest, "audit_directory": audit.as_posix(),
        "scope": "Small verified report exhibits; raw data, responses and weights are retained separately.",
        "new_inference_calls": 0, "actual_provider_bill_reconciled": False})
    print(f"Published {len(manifest)} verified exhibits to {target}; no original artifact modified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory; existing results cannot be overwritten")
    args = parser.parse_args()
    publish(args.audit_dir, args.output_dir)


if __name__ == "__main__":
    main()
