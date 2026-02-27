import torch
import random
import matplotlib.pyplot as plt
from utils.model_persistence.load import load_model
from utils.proj_tools.helpers import cmd_cleaner
from models.core_model import AnimalsVisionModelV0
from torchvision.datasets import ImageFolder
from data_setup.transform import get_transform
from config.settings import TEST_DIR, MODEL_SAVE_PATH

cmd_cleaner(clean=True, print_cleaning=False)

random.seed(32)
# torch.manual_seed(32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = get_transform(mood="test", img_size=128)


mod = AnimalsVisionModelV0(num_classes=3, in_channels=3)
model = load_model(model=mod, model_path=MODEL_SAVE_PATH, device=device, strict=True)

dataset = ImageFolder(
    root=TEST_DIR,
    transform=transform,
)

import torchvision

datasetnew = ImageFolder(
    root=TEST_DIR,
    transform=torchvision.transforms.ToTensor()
)

model = model.to(device)


def make_predictions(model: torch.nn.Module,
                     data,
                     device: torch.device = device):

    pred_probs = []

    model.eval()
    with torch.inference_mode():

        for image, _ in data:   
            image = image.unsqueeze(0).to(device)

            pred_logit = model(image)

            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)


pred_probs = make_predictions(model=model,
                              data=dataset)

pred_classes = pred_probs.argmax(dim=1)


class_names = dataset.classes  # مهم جداً

plt.figure(figsize=(9, 9))

nrows = 3
ncols = 3

for i in range(9):

    random_idx = random.randint(0, len(dataset) - 1)
    
    image, label = dataset[random_idx]

    plt.subplot(nrows, ncols, i + 1)

    plt.imshow(image.squeeze().permute(1, 2, 0), cmap="gray")

    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[label]

    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="green")
    else:
        plt.title(title_text, fontsize=10, c="red")

    plt.axis(False)

plt.tight_layout()
plt.show()

rim = random.randint(0, len(datasetnew) -1)
img_, _ = datasetnew[rim]

print(img_)
plt.imshow(img_.squeeze(dim=0).permute(1, 2, 0))
plt.show()

transformss = torchvision.transforms.Compose(
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
)

img__ =  transformss(img_)
plt.imshow(img__.squeeze(dim=0).permute(1, 2, 0))
plt.show()


import random
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

# Function to convert Tensor to cv2-friendly numpy image
def tensor_to_cv2(img_tensor, normalize=False):
    img_np = img_tensor.squeeze(0).numpy()  # [1,H,W] -> [H,W]
    if normalize:
        img_np = ((img_np + 1) * 127.5).astype(np.uint8)  # [-1,1] -> [0,255]
    else:
        img_np = (img_np * 255).astype(np.uint8)  # [0,1] -> [0,255]
    return img_np

# Randomly select an image
rim = random.randint(0, len(datasetnew) - 1)
img_, _ = datasetnew[rim]  # img_ is Tensor [C,H,W]

# Show original image
img_cv = tensor_to_cv2(img_)
cv2.imshow("Original Image", img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Transform: Grayscale + Normalize
transformss = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

img__ = transformss(img_)

# Show normalized image
img_cv2 = tensor_to_cv2(img__, normalize=True)
cv2.imshow("Grayscale + Normalize", img_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Canny edge detection
edges = cv2.Canny(img_cv2, 100, 200)
cv2.imshow("Edges (Canny)", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()