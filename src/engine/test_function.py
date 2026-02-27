import torch 
from torch import nn
import torchmetrics
import pandas as pd
from tqdm.auto import tqdm

def test_step(model: nn.Module,
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
    test_loss = 0
    test_loss_list = []
    test_acc_list = []
    test_results_list = []
    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(dataloader)):
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

            test_loss += loss.item()
            
            accuracy_fn.update(labels, y)
            test_acc_list.append(accuracy_fn.compute().item())
            
            if batch % print_each == 0:
                print(f"\nEpoch: {epoch} | Batch: {batch}\nLoss: {loss.item():.4f} | Accuracy: {accuracy_fn.compute().item()*100:.2f}%\nf-1 score: {f1_score.compute().item()}")
                test_results_list.append({
                    "loop_task": "test",
                    "epoch": epoch,
                    "batch": batch,
                    "loss": loss.item(),
                    "accuracy": accuracy_fn.compute().item()
                })
                writer.add_scalar(f"Loss/Test", loss.item(), epoch * len(dataloader) + batch)
                writer.add_scalar(f"Accuracy/Test", accuracy_fn.compute().item(), epoch * len(dataloader) + batch)
                
    test_loss_list.append(loss.item())
    test_acc_list.append(accuracy_fn.compute().item())
    if f1_score is not None:
        f1_test = f1_score.compute().item()
    test_loss /= len(dataloader)
    test_acc = accuracy_fn.compute().item()
    test_results = pd.DataFrame(test_results_list)
    accuracy_fn.reset()

    if print_last_result:
        print(f"Epoch: {epoch}\n| Average Loss: {test_loss:.2f} | Average Accuracy: {test_acc:.4f}%")
    if return_results:
        if f1_score is not None:
            return test_loss, test_acc, test_loss_list, test_acc_list, test_results, f1_test
        else:
            return test_loss, test_acc, test_loss_list, test_acc_list, test_results
