from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

def detailed_mismatch_report(images_dir: Path, labels_dir: Path) -> None:
    if not images_dir.exists() or not labels_dir.exists():
        print("Error: 'images' or 'labels' directory is missing.")
        return

    images_without_label = []
    orphan_labels = []
    empty_labels = []
    valid_pairs = 0

    # 1. Check all images
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            images_without_label.append(img_path.name)
            continue

        content = label_path.read_text(encoding="utf-8").strip()
        if content == "":
            empty_labels.append(img_path.name)   # show image name (same stem)
            continue

        valid_pairs += 1

    # 2. Check for labels without corresponding image
    for label_path in sorted(labels_dir.iterdir()):
        if not label_path.is_file() or label_path.suffix.lower() != ".txt":
            continue
        if label_path.name == "classes.txt":
            continue

        has_image = any(
            (images_dir / f"{label_path.stem}{ext}").exists()
            for ext in IMAGE_EXTENSIONS
        )
        if not has_image:
            orphan_labels.append(label_path.name)

    # Final detailed report
    print("\n" + "="*60)
    print("           DATASET DETAILED HEALTH REPORT")
    print("="*60)
    print(f"Valid image-label pairs            : {valid_pairs}")
    print(f"Images WITHOUT label               : {len(images_without_label)}")
    print(f"Labels WITHOUT image               : {len(orphan_labels)}")
    print(f"Empty label files                  : {len(empty_labels)}")
    print("-"*60)

    if images_without_label:
        print(f"\nImages without corresponding .txt label ({len(images_without_label)}):")
        for name in images_without_label:
            print(f"   → {name}")

    if empty_labels:
        print(f"\nEmpty label files (image exists but .txt is empty) ({len(empty_labels)}):")
        for name in empty_labels:
            print(f"   → {name}  +  {name.rsplit('.', 1)[0]}.txt (empty)")

    if orphan_labels:
        print(f"\nLabel files without any image ({len(orphan_labels)}):")
        for name in orphan_labels:
            print(f"   → {name}")

    if not images_without_label and not orphan_labels and not empty_labels:
        print("\nPerfect! All images and labels are perfectly matched and none are empty.")

    print("\nNo files were modified, moved, or deleted.")
    print("="*60)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"

    detailed_mismatch_report(images_dir, labels_dir)