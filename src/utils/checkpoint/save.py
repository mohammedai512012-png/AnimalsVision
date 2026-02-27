import torch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str, scheduler = None, if_print_results:  bool = False):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None
    }

    torch.save(checkpoint, path)
    if if_print_results:
        print(f"Checkpoint saved to {path}")
        