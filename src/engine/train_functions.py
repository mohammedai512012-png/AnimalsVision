import torch 
from torch import nn, optim
import torchmetrics
import pandas as pd

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: optim.Optimizer,
               accuracy_fn: torchmetrics.Accuracy,
               writer,
               task: str,
               epoch: int,
               print_each: int,
               f1_score: torchmetrics.F1Score = None,
               print_last_result: bool = False,
               return_results: bool = False,
               inside_scheduler=None,
               device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    if f1_score is not None:
        f1_score.to(device)
        f1_score.reset()
    model.train()
    accuracy_fn.reset()
    train_loss = 0
    train_loss_list = []
    train_acc_list = []
    train_results_list = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)

        if task == "binary":
            labels = (torch.sigmoid(pred) > 0.5).int()
        elif task == "multi-class":
            labels = torch.argmax(pred, dim=1)
        else:
            raise ValueError("Invalid argument: task must be 'binary' or 'multi-class'.")
        loss =loss_fn(pred, y)

        if f1_score is not None:
            f1_score.update(labels, y)

        train_loss += loss.item()
        accuracy_fn.update(labels, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if inside_scheduler is not None:
            inside_scheduler.step()
        
        if batch % print_each == 0:
            print(f"Epoch: {epoch} | Batch: {batch}\nLoss: {loss.item():.4f} | Accuracy: {accuracy_fn.compute().item()*100:.2f}%\nF-1 score: {f1_score.compute().item()}")
            train_results_list.append({
                "loop_task": "train",
                "epoch": epoch,
                "batch": batch,
                "loss": loss.item(),
                "accuracy": accuracy_fn.compute().item()
            })
            writer.add_scalar(f"Loss/Train", loss.item(), epoch * len(dataloader) + batch)
            writer.add_scalar(f"Accuracy/Train", accuracy_fn.compute().item(), epoch * len(dataloader) + batch)
    
    train_acc_list.append(accuracy_fn.compute().item())
    train_loss_list.append(loss.item())
    if f1_score is not None:
        train_f1 = f1_score.compute().item()
    train_loss /= len(dataloader)
    train_acc = accuracy_fn.compute().item()
    train_results = pd.DataFrame(train_results_list)
    accuracy_fn.reset()

    if print_last_result:
        print(f"\nEpoch: {epoch}\n| Average Loss: {train_loss:.2f} | Average Accuracy: {train_acc:.4f}%\nF1-score: {f1_score.compute().item():.4f}")
    if return_results:
        if f1_score is not None:
            return train_loss, train_acc, train_loss_list, train_acc_list, train_results, train_f1
        else:
            return train_loss, train_acc, train_loss_list, train_acc_list, train_results
        