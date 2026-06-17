from pathlib import Path
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)

    if not root.is_dir():
        raise ValueError(f"{root=} is not a directory")

    has_checkpoints = []
    no_checkpoints = []

    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue

        if (subdir / "checkpoints").is_dir():
            has_checkpoints.append(subdir)
        else:
            no_checkpoints.append(subdir)

    print(f"\nFolders WITH checkpoints ({len(has_checkpoints)}):")
    for d in sorted(has_checkpoints):
        print(f"  KEEP   {d}")

    print(f"\nFolders WITHOUT checkpoints ({len(no_checkpoints)}):")
    for d in sorted(no_checkpoints):
        print(f"  DELETE {d}")

    if not args.delete:
        print("\nDRY RUN ONLY. Nothing deleted.")
        return

    print("\nDeleting folders without checkpoints...")
    for d in no_checkpoints:
        shutil.rmtree(d)
        print(f"Deleted: {d}")

if __name__ == "__main__":
    main()