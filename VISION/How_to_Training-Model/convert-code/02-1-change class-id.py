from pathlib import Path


def change_class_ids(labels_dir: Path) -> None:
    if not labels_dir.exists():
        print("Labels directory not found.")
        return

    total_files = 0
    updated_files = 0

    for label_file in sorted(labels_dir.glob("*.txt")):
        if label_file.name == "classes.txt":
            continue

        total_files += 1
        text = label_file.read_text(encoding="utf-8")
        lines = text.splitlines()
        newline = "\n" if text.endswith("\n") else ""

        new_lines = []
        changed = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                new_lines.append(line)
                continue

            parts = stripped.split()
            if parts[0] == "1":
                parts[0] = "5"
                new_lines.append(" ".join(parts))
                changed = True
            else:
                new_lines.append(line)

        if changed:
            updated_files += 1
            label_file.write_text("\n".join(new_lines) + newline, encoding="utf-8")

    print(f"Checked {total_files} label files.")
    print(f"Updated {updated_files} files where class-id 1 was changed to 5.")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    labels_dir = base_dir / "labels"
    change_class_ids(labels_dir)

