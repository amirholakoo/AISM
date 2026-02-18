from pathlib import Path
import shutil

IMAGE_EXTENSIONS = {".png"}


def remove_empty_pairs(images_dir: Path, labels_dir: Path) -> None:
    if not images_dir.exists() or not labels_dir.exists():
        print("Images or labels directory is missing.")
        return

    tmp_images = images_dir.parent / "__tmp_images__"
    tmp_labels = labels_dir.parent / "__tmp_labels__"
    if tmp_images.exists():
        shutil.rmtree(tmp_images)
    if tmp_labels.exists():
        shutil.rmtree(tmp_labels)
    tmp_images.mkdir()
    tmp_labels.mkdir()

    total = kept = removed = 0
    valid_pairs = []

    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total += 1
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            if img_path.exists():
                img_path.unlink()
            removed += 1
            continue

        label_content = label_path.read_text(encoding="utf-8").strip()
        if not label_content:
            if label_path.exists():
                label_path.unlink()
            if img_path.exists():
                img_path.unlink()
            removed += 1
            continue

        valid_pairs.append((img_path, label_path))
        kept += 1

    for idx, (img_path, label_path) in enumerate(valid_pairs):
        new_stem = f"frame_{idx:06d}"
        new_img = tmp_images / f"{new_stem}{img_path.suffix.lower()}"
        new_label = tmp_labels / f"{new_stem}.txt"
        shutil.move(str(img_path), new_img)
        shutil.move(str(label_path), new_label)

    classes_file = labels_dir / "classes.txt"
    if classes_file.exists():
        shutil.copy2(classes_file, tmp_labels / "classes.txt")

    shutil.rmtree(images_dir)
    shutil.rmtree(labels_dir)
    tmp_images.rename(images_dir)
    tmp_labels.rename(labels_dir)

    print(f"Checked {total} images.")
    print(f"Kept {kept} pairs, removed {removed}.")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    remove_empty_pairs(images_dir, labels_dir)

