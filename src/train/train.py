from src.config.settings import MODEL_SAVE_PATH, CHECKPOINT_SAVE_PATH, LOGS_PATH
from src.config.settings import NUM_EPOCHS, PRINT_EACH, LR
from src.config.settings import __name__, __version__
from train.data_setupRun import train_data
from train.data_setupRun import train_dataloader, eval_dataloader
from src.engine.train_functions import train_step
from src.engine.eval_function import eval_step
from src.models.core_model import AnimalsVisionModelV0
from src.utils.model_persistence.save import save_model
from src.utils.proj_tools.helpers import cmd_cleaner, find_device, set_seed, timer_count
from src.utils.checkpoint.save import save_checkpoint
from src.utils.checkpoint.load import load_checkpoint
from src.utils.curves.curves import plot_curves

import torch
import torchvision
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
from tqdm.auto import tqdm

cmd_cleaner(clean=True)
set_seed(seed=42, print_seed=True)

print(f"model: {__name__} | version: {__version__}\n",
      f"PyTorch: {torch.__version__} | torchvision: {torchvision.__version__} | torchmetrics: {torchmetrics.__version__}")

device = find_device(print_device=True)
model = AnimalsVisionModelV0(
    output_shape=len(train_dataloader.dataset.class_names)
    ).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=NUM_EPOCHS, gamma=0.8)
accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=len(train_dataloader.dataset.class_names)).to(device)
f1_score = torchmetrics.F1Score(task='multiclass', num_classes=len(train_data.class_names), average='macro')
writer = SummaryWriter(log_dir=LOGS_PATH)

train_loss_list, train_acc_list, eval_loss_list, eval_acc_list = [], [], [], []

start_time = timer()
for epoch in tqdm(range(NUM_EPOCHS)):
    tqdm.write(f"\nEpoch: {epoch}\n-------------")
    train_loss, train_acc, _, _, train_results, train_f1_score = train_step(
        model=model,
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        writer=writer,
        task="multi-class",
        epoch=epoch,
        f1_score=f1_score,
        print_each=PRINT_EACH,
        inside_scheduler=None,
        return_results=True,
        device=device
    )

    eval_loss, eval_acc, _, _, eval_results, eval_f1_score = eval_step(
        model=model,
        dataloader=eval_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        writer=writer,
        task="multi-class",
        epoch=epoch,
        f1_score=f1_score,
        print_each=PRINT_EACH,
        return_results=True,    
        device=device
    )
    
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    eval_loss_list.append(eval_loss)
    eval_acc_list.append(eval_acc)

    checkpoint_path = CHECKPOINT_SAVE_PATH / f"checkpoint_epoch_{epoch+2:2d}.pth"
    save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            scheduler=scheduler,
            path=checkpoint_path,
            if_print_results=True
        )
    
    scheduler.step()

writer.close()

end_time = timer()

MODEL_SAVE_PATH = MODEL_SAVE_PATH
save_model(model=model, path=MODEL_SAVE_PATH, print_save_results=True)
timer_count(start_time, end_time, print_time=True)

epochs = list(range(1, NUM_EPOCHS+1))

plot_curves(train_loss_list=train_loss_list,
            train_acc_list=train_acc_list,
            eval_loss_list=eval_loss_list,
            eval_acc_list=eval_acc_list,
            epochs=epochs)
