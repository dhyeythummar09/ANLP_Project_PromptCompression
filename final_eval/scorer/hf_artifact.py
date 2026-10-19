"""Inspect or explicitly download the public legacy custom checkpoint bundle."""
import argparse
from pathlib import Path
from final_eval.utils.artifacts import atomic_json, claim_output

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="flashstep/reasoning-critical-span-scorer")
    ap.add_argument("--revision", default="c24d8ce9ef2cfbdc7b11185958dee5810327e432")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--download", action="store_true", help="Explicitly download the five weight files; run on server")
    args = ap.parse_args()
    claim_output(args.output_dir)
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(args.repo, revision=args.revision, token=False)
    target = Path(args.output_dir); target.mkdir(parents=True)
    atomic_json(target / "artifact_metadata.json", {"repo": args.repo, "revision": info.sha,
        "files": [f.rfilename for f in info.siblings], "target_kind": "unknown_from_public_metadata",
        "split_safety": "historical; must audit base families before scientific reuse"})
    if args.download:
        snapshot_download(args.repo, revision=info.sha, token=False, local_dir=target,
                          allow_patterns=["README.md", "split.json", "fold*/scorer.pt"])
    print(f"Saved artifact metadata to {target}; weights downloaded: {args.download}")

if __name__ == "__main__":
    main()
