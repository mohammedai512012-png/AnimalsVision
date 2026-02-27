import torch
from colorama import Fore, init

def save_model(model: torch.nn.Module, path: str, print_save_results: bool = True):
    torch.save(model.state_dict(), path)
    if print_save_results:
        print(f"Model saved to {Fore.LIGHTGREEN_EX}{path}{Fore.RESET}")

# def load_model(model: torch.nn.Module, path: str, device: torch.device = None, print_load_results: bool = True):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     model.load_state_dict(torch.load(path, map_location=device))
#     model.to(device)
#     if print_load_results:
#         print(f"Model loaded from {Fore.LIGHTGREEN_EX}{path}{Fore.RESET} to {device}")
