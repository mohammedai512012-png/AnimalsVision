import torch
from colorama import Fore, init
from PIL import Image
import io

init(autoreset=True)

def cmd_cleaner(clean: bool = False, print_cleaning: bool = True):
    if clean:
        import os
        os.system("cls" if os.name == "nt" else "clear")
        if print_cleaning:
            print(f"{Fore.LIGHTGREEN_EX}Command line cleaned!{Fore.RESET}")

def find_device(print_device: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if print_device:
        print(f"Using {Fore.LIGHTGREEN_EX}{device}{Fore.RESET} device.")
    return device

def set_seed(seed: int = 42, print_seed: bool = True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    if print_seed:
        print(f"Random seed set to {Fore.LIGHTGREEN_EX}{seed}{Fore.RESET} for reproducibility.")

def timer_count(start: float, end: float, print_time: bool = False):
    elapsed_time = end - start
    if print_time:
        print(f"Elapsed time: {Fore.LIGHTGREEN_EX}{elapsed_time:.2f} seconds{Fore.RESET}")
    return elapsed_time

def infernece(model: torch.nn.Module, class_names, image_path: str, transform, device: torch.device = None):
    if device is None:
        device = find_device(print_device=True)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        pred = model(image_tensor)
        pred_label = class_names[torch.argmax(pred, dim=1).item()]
        
    return pred_label

import io

def inference(model: torch.nn.Module, class_names, file_bytes, transform, device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    model.to(device)
    model.eval()

    with torch.inference_mode():
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        pred = model(image_tensor)
        pred_label = class_names[torch.argmax(pred, dim=1).item()]

    return pred_label
