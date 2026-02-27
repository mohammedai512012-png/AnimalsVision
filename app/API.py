# from src.models.core_model import AnimalsVisionModelV0
# from src.data_setup.transform import get_transform
# from src.utils.helper_functions.func.load import load_model
# from src.utils.helper_functions.proj_tools.utils import cmd_cleaner, find_device, set_seed, infernece
# from src.run.data_setupRun import eval_data

# import torch
# from PIL import Image
# import io

# class_names = eval_data.class_names

# cmd_cleaner(False)

# eval_transform = get_transform(mood="val", img_size=128)

# device = find_device(print_device=True)
# set_seed(seed=42, print_seed=True)
# model_shape = AnimalsVisionModelV0(output_shape=3).to(device)
# model = load_model(model=model_shape, model_path="C:/Users/Mohamed/Desktop/AnimalsVision/models/animalsvisionV0.pth")
# load_model(model=model, model_path="C:/Users/Mohamed/Desktop/AnimalsVision/models/animalsvisionV0.pth", device=device)

# Image_Path = "otherdata/val/pizza/398345.jpg"

# predic = infernece(model=model, image_path=Image_Path, class_names=class_names, transform=eval_transform, device=device)

# print(f"model prediction: {predic}")

# from src.models.core_model import AnimalsVisionModelV0
# from src.utils.helper_functions.proj_tools.utils import inference

# class_names = eval_data.class_names

# predict_transform = get_transform(mood="val", img_size=128)


 
# def inference(model: torch.nn.Module, class_names, file_bytes, transform, device: torch.device = None):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {device}")

#     model.to(device)
#     model.eval()

#     with torch.inference_mode():
#         image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
#         image_tensor = transform(image).unsqueeze(0).to(device)

#         pred = model(image_tensor)
#         pred_label = class_names[torch.argmax(pred, dim=1).item()]

#     return pred_label

# def web_inference(image_path):
#     pred = inference(model=model, class_names=class_names, file_bytes=image_path, transform=eval_transform, device=device)
    
#     return pred

# if __name__ == '__main__':
#     print(f"[INFO] From API.py")
