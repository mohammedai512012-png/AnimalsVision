from src.data_setup.dataset import AnimalsDataset
from src.data_setup.transform import get_transform
from src.data_setup.dataloader import get_dataloader

from src.config.settings import TRAIN_DIR, VAL_DIR
from src.config.settings import NUM_WORKERS, BATCH_SIZE

import os

NUM_WORKERS = NUM_WORKERS if NUM_WORKERS != "autoscala" else os.cpu_count()

train_transform = get_transform(
    mood="train",
    img_size=128
)

eval_transform = get_transform(
    mood="test",
    img_size=128
)

train_data = AnimalsDataset(
    root_dir=TRAIN_DIR,
    transform=train_transform
)

eval_data = AnimalsDataset(
    root_dir=VAL_DIR,
    transform=eval_transform
)

train_dataloader = get_dataloader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

eval_dataloader = get_dataloader(
    dataset=eval_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)
