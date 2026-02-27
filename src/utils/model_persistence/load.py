import torch
from pathlib import Path


def load_model(
    model: torch.nn.Module,
    model_path: str,
    device: torch.device = None,
) -> torch.nn.Module:
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)


    return model
