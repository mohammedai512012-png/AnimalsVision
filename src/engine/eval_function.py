import torch 
from torch import nn
import torchmetrics
import pandas as pd

def eval_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               accuracy_fn: torchmetrics.Accuracy,
               writer,
               task: str,
               epoch: int,
               print_each: int,
               f1_score: torchmetrics.F1Score = None,
               print_last_result: bool = False,
               return_results: bool = False,
               device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    if f1_score is not None:
        f1_score.to(device)
        f1_score.reset()
    model.eval()
    accuracy_fn.reset()
    eval_loss = 0
    eval_loss_list = []
    eval_acc_list = []
    eval_results_list = []
    with torch.inference_mode():
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

            eval_loss += loss.item()
            accuracy_fn.update(labels, y)
            
            if batch % print_each == 0:
                print(f"\nEpoch: {epoch} | Batch: {batch}\nLoss: {loss.item():.4f} | Accuracy: {accuracy_fn.compute().item()*100:.2f}%\nF1-score: {f1_score.compute().item():.4f}")
                eval_results_list.append({
                    "loop_task": "eval",
                    "epoch": epoch,
                    "batch": batch,
                    "loss": loss.item(),
                    "accuracy": accuracy_fn.compute().item()
                })
                writer.add_scalar(f"Loss/Eval", loss.item(), epoch * len(dataloader) + batch)
                writer.add_scalar(f"Accuracy/Eval", accuracy_fn.compute().item(), epoch * len(dataloader) + batch)
    eval_loss_list.append(loss.item())
    eval_acc_list.append(accuracy_fn.compute().item())
    if f1_score is not None:
        f1_eval = f1_score.compute().item()
    eval_loss /= len(dataloader)
    eval_acc = accuracy_fn.compute().item()
    eval_results = pd.DataFrame(eval_results_list)
    accuracy_fn.reset()

    if print_last_result:
        print(f"Epoch: {epoch}\n| Average Loss: {eval_loss:.2f} | Average Accuracy: {eval_acc:.4f}%")
    if return_results:
        if f1_score:
            return eval_loss, eval_acc,eval_loss_list, eval_acc_list, eval_results, f1_eval
        else:
            return eval_loss, eval_acc,eval_loss_list, eval_acc_list, eval_results
    