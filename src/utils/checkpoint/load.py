import torch

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, path: str, device: torch.device = None, print_load_results: bool = True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    
    model.to(device)
    
    if print_load_results:
        print(f"Checkpoint loaded from {path} to {device} at epoch {epoch}")
    
    return epoch
