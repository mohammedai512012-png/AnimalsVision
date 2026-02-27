import os
import random
import shutil
import yaml

config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs/paths.yml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

source_dir = config["data_dirs"]["source_data"]

target_dir = config["data_dirs"]["target_data"]

config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs/data_setup.yml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

train_ratio = config["data_split"]["train_ratio"]
val_ratio = config["data_split"]["val_ratio"]
test_ratio = config["data_split"]["test_ratio"]

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(target_dir, split), exist_ok=True)

classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

for cls in classes:
    cls_source = os.path.join(source_dir, cls)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

    files = [f for f in os.listdir(cls_source) if os.path.isfile(os.path.join(cls_source, f))]
    random.shuffle(files)

    n_total = len(files)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val  

    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]

    for f in train_files:
        shutil.move(os.path.join(cls_source, f), os.path.join(target_dir, "train", cls, f))
    for f in val_files:
        shutil.move(os.path.join(cls_source, f), os.path.join(target_dir, "val", cls, f))
    for f in test_files:
        shutil.move(os.path.join(cls_source, f), os.path.join(target_dir, "test", cls, f))

print("Done! Data split into train/val/test in 'data/' folder.")
