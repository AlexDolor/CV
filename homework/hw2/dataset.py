from pathlib import Path
from collections import Counter
from tqdm import tqdm
from collections import defaultdict
import random
import yaml
import shutil

def count_classes(path):
    labels_dir = Path(path)
    counter = Counter()

    for txt in tqdm(labels_dir.glob("*.txt")):
        with open(txt, "r") as f:
            for line in f:
                cls_id = int(line.split()[0])
                counter[cls_id] += 1
    return counter

def check_classes():
    common_path = 'homework/hw2/dataset/dogs_cats_and_birds' 
    path = common_path + "/train/labels"
    print('train classes', count_classes(path))

    path = common_path + "/test/labels"
    print('test classes', count_classes(path))

    path = common_path + "/valid/labels"
    print('val classes', count_classes(path))

# check_classes()

#SETTINGS

SOURCE_ROOT = Path("homework/hw2/dataset/dogs_cats_and_birds") 
TARGET_ROOT = Path("homework/hw2/dataset")          

TARGET_COUNTS = {
    0: 100,  # cat
    1: 100,  # dog
    2: 100,  # bird
}

RANDOM_SEED = 42

def load_class_names(data_yaml_path: Path):
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        max_idx = max(int(k) for k in names.keys())
        name_list = [""] * (max_idx + 1)
        for k, v in names.items():
            name_list[int(k)] = v
        return name_list, data
    else:
        return names, data


def parse_label_file(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            rest = " ".join(parts[1:])
            boxes.append((cls_id, rest))
    return boxes


def write_label_file(label_path: Path, boxes):
    with open(label_path, "w") as f:
        for cls_id, rest in boxes:
            f.write(f"{cls_id} {rest}\n")


def copy_image_and_label(src_img: Path, src_lbl: Path,
                         dst_img: Path, dst_lbl: Path,
                         selected_cls_ids=None):
    shutil.copy2(src_img, dst_img)
    boxes = parse_label_file(src_lbl)
    if selected_cls_ids is not None:
        boxes = [b for b in boxes if b[0] in selected_cls_ids]
    if boxes:
        write_label_file(dst_lbl, boxes)
    elif dst_lbl.exists():
        dst_lbl.unlink()


def analyze_train_split(train_img_dir: Path, train_lbl_dir: Path):
    train_images = sorted(
        list(train_img_dir.glob("*.jpg")) +
        list(train_img_dir.glob("*.jpeg")) +
        list(train_img_dir.glob("*.png"))
    )

    image_objects = []  # (img_path, lbl_path, Counter по классам)
    global_counts = defaultdict(int)

    for img_path in train_images:
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        boxes = parse_label_file(lbl_path)
        if not boxes:
            continue
        cls_counter = defaultdict(int)
        for cls_id, _ in boxes:
            cls_counter[cls_id] += 1
            global_counts[cls_id] += 1
        image_objects.append((img_path, lbl_path, cls_counter))

    return image_objects, global_counts


def select_train_subset(image_objects, target_counts: dict):
    '''select images from train folder
    Number of selected images in each class is determined by target_counts'''
    random.shuffle(image_objects)
    selected_images = []
    current_counts = defaultdict(int)

    for img_path, lbl_path, cls_counter in image_objects:
        # проверяем, добавит ли картинка что-то полезное
        can_use = any(
            (cls_id in target_counts and current_counts[cls_id] < target_counts[cls_id])
            for cls_id in cls_counter.keys()
        )
        if not can_use:
            continue

        add_cls_ids = set()
        for cls_id, _ in cls_counter.items():
            if cls_id not in target_counts:
                continue
            if current_counts[cls_id] >= target_counts[cls_id]:
                continue
            add_cls_ids.add(cls_id)

        if not add_cls_ids:
            continue

        for cls_id, n in cls_counter.items():
            if cls_id in add_cls_ids:
                current_counts[cls_id] += n

        selected_images.append((img_path, lbl_path, add_cls_ids))

        if all(current_counts[c] >= target_counts[c] for c in target_counts):
            break

    return selected_images, current_counts


def ensure_target_dirs(target_root: Path):
    for split in ["train", "valid", "test"]:
        (target_root / split / "images").mkdir(parents=True, exist_ok=True)
        (target_root / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_full_split(source_root: Path, target_root: Path, split_name: str):
    src_split = source_root / split_name
    if not src_split.exists():
        return

    src_img_dir = src_split / "images"
    src_lbl_dir = src_split / "labels"

    dst_img_dir = target_root / split_name / "images"
    dst_lbl_dir = target_root / split_name / "labels"

    imgs = sorted(
        list(src_img_dir.glob("*.jpg")) +
        list(src_img_dir.glob("*.jpeg")) +
        list(src_img_dir.glob("*.png"))
    )
    for img_path in imgs:
        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
        dst_img = dst_img_dir / img_path.name
        dst_lbl = dst_lbl_dir / lbl_path.name
        shutil.copy2(img_path, dst_img)
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl)


def write_new_data_yaml(target_root: Path, class_names, original_yaml: dict):
    new_data = {
        "path": str(target_root),
        "train": "train/images",
        "val": "valid/images",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    # добавим test, если он есть
    if (target_root / "test" / "images").exists():
        new_data["test"] = "test/images"

    # можно при желании перенести другие поля из оригинального YAML
    with open(target_root / "data.yaml", "w") as f:
        yaml.dump(new_data, f)


# ===== MAIN =====

def main():
    random.seed(RANDOM_SEED)

    data_yaml_path = SOURCE_ROOT / "data.yaml"
    class_names, data_yaml = load_class_names(data_yaml_path)
    print("Классы в исходном датасете:", class_names)

    train_img_dir = SOURCE_ROOT / "train" / "images"
    train_lbl_dir = SOURCE_ROOT / "train" / "labels"

    # анализируем train
    image_objects, global_counts = analyze_train_split(train_img_dir, train_lbl_dir)
    print("Всего объектов по классам в train:")
    for cls_id, cnt in global_counts.items():
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        print(f"  {cls_id} ({name}): {cnt}")

    # выбираем подмножество train
    selected_images, current_counts = select_train_subset(image_objects, TARGET_COUNTS)
    print("Выбрано изображений для train:", len(selected_images))
    print("Выбрано объектов по классам:")
    for cls_id, cnt in current_counts.items():
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        print(f"  {cls_id} ({name}): {cnt}")

    # создаём директории
    ensure_target_dirs(TARGET_ROOT)

    # копируем train (только выбранные)
    target_train_img_dir = TARGET_ROOT / "train" / "images"
    target_train_lbl_dir = TARGET_ROOT / "train" / "labels"
    for img_path, lbl_path, add_cls_ids in selected_images:
        dst_img = target_train_img_dir / img_path.name
        dst_lbl = target_train_lbl_dir / lbl_path.name
        copy_image_and_label(img_path, lbl_path, dst_img, dst_lbl, selected_cls_ids=add_cls_ids)

    # копируем valid и test целиком
    copy_full_split(SOURCE_ROOT, TARGET_ROOT, "valid")
    copy_full_split(SOURCE_ROOT, TARGET_ROOT, "test")

    # пишем новый data.yaml
    write_new_data_yaml(TARGET_ROOT, class_names, data_yaml)

    print("Готово. Итоговый датасет в папке:", TARGET_ROOT)


if __name__ == "__main__":
    main()