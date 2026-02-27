from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT_DIR / "configs" / "paths.yml"
DATA_SETUP_PATH = ROOT_DIR / "configs" / "data_setup.yml"
TRAIN_PATH = ROOT_DIR / "configs" / "train.yml"
PROJECT_SETTINGS = ROOT_DIR / "configs" / "_project.yml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

with open(DATA_SETUP_PATH, "r") as f:
    data_setup = yaml.safe_load(f)

with open(TRAIN_PATH, "r") as f:
    train = yaml.safe_load(f)

with open(PROJECT_SETTINGS, "r") as f: 
    proj_settings = yaml.safe_load(f)

SOURCE_DATA_DIR = ROOT_DIR / config["data_dirs"]["source_data"]
TARGET_DATA_DIR = ROOT_DIR / config["data_dirs"]["target_data"]

MODEL_SAVE_PATH = ROOT_DIR / config["saved_model_dirs"]["model_save_path"]
CHECKPOINT_SAVE_PATH = ROOT_DIR /config["saved_checkpoint_dirs"]["checkpoint_save_path"]
LOGS_PATH = ROOT_DIR / config["saved_logs_dir"]["logs_path"]

TRAIN_DIR = TARGET_DATA_DIR / "train"
TEST_DIR = TARGET_DATA_DIR / "val"

VAL_DIR = TARGET_DATA_DIR / "val"

paths = [SOURCE_DATA_DIR,
         TARGET_DATA_DIR]

for path in paths:
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] {paths[path]} not found.")

BATCH_SIZE = data_setup["data_loader"]["batch_size"]
IMG_SIZE = data_setup["data_transforms"]["img_size"]

NUM_EPOCHS = train["train"]["epochs"]
LR = train["train"]["LR"]
PRINT_EACH = train["train"]["print_each"]
NUM_WORKERS = data_setup["data_transforms"]["num_workers"]

__version__ = proj_settings["info"]["__version__"]
__name__ = proj_settings["info"]["__name__"]
__status__ = proj_settings["info"]["__status__"]
