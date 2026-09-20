import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

!pip -q install torch torchvision matplotlib pandas pillow

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import datasets, transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    ViT_B_16_Weights
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

"""## fixed the random seed so that the experiment is more reproducible"""

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Seed:", SEED)

"""## Loading models

"""

# -------------------------
# ResNet50
# -------------------------
resnet_weights = ResNet50_Weights.DEFAULT
resnet_model = resnet50(weights=resnet_weights)
resnet_model = resnet_model.to(device)
resnet_model.eval()

# -------------------------
# ViT-B/16
# -------------------------
vit_weights = ViT_B_16_Weights.DEFAULT
vit_model = vit_b_16(weights=vit_weights)
vit_model = vit_model.to(device)
vit_model.eval()

print("ResNet50 loaded")
print("ViT-B/16 loaded")

# model info
print("ResNet50 parameters:",
      sum(p.numel() for p in resnet_model.parameters()))

print("ViT-B/16 parameters:",
      sum(p.numel() for p in vit_model.parameters()))

# preprocessing

resnet_transform = resnet_weights.transforms()
vit_transform = vit_weights.transforms()

print(resnet_transform)
print(vit_transform)

"""## DATASET"""

from torchvision import datasets

raw_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True
)

print("Total test images:", len(raw_dataset))
print("Classes:", raw_dataset.classes)

# all cat images
CAT_CLASS_ID = raw_dataset.class_to_idx["cat"]

cat_indices = [
    i for i, label in enumerate(raw_dataset.targets)
    if label == CAT_CLASS_ID
]

print("Number of cat images:", len(cat_indices))
print("Cat class ID:", CAT_CLASS_ID)

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

class CIFARCatDataset(torch.utils.data.Dataset):

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        image, label = self.base_dataset[real_idx]

        image = self.transform(image)

        return image, label, real_idx

from torchvision import transforms

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# dataset for the models

resnet_cat_dataset = CIFARCatDataset(
    raw_dataset,
    cat_indices,
    resnet_transform
)

vit_cat_dataset = CIFARCatDataset(
    raw_dataset,
    cat_indices,
    vit_transform
)

print("ResNet dataset:", len(resnet_cat_dataset))
print("ViT dataset:", len(vit_cat_dataset))

from torchvision import models

resnet_weights = models.ResNet50_Weights.DEFAULT

# getiing imagenet class names -
imagenet_categories = resnet_weights.meta["categories"]

print("Number of ImageNet classes:", len(imagenet_categories))

for i, name in enumerate(imagenet_categories):
    if "cat" in name.lower():
        print(i, name)


# collecting them

CAT_IMAGENET_IDS = [
    i for i, name in enumerate(imagenet_categories)
    if "cat" in name.lower()
]

CAT_IMAGENET_NAMES = [
    imagenet_categories[i]
    for i in CAT_IMAGENET_IDS
]

print("Cat-related ImageNet classes:")
for i, name in zip(CAT_IMAGENET_IDS, CAT_IMAGENET_NAMES):
    print(i, name)

"""# Prediction

"""

@torch.no_grad()
def predict(model, images, categories):
    """
    Returns top-1 ImageNet prediction and confidence.
    """

    outputs = model(images)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = probabilities.max(dim=1)

    results = []

    for i in range(len(images)):
        class_id = predicted[i].item()
        conf = confidence[i].item()

        results.append({
            "class_id": class_id,
            "class_name": categories[class_id],
            "confidence": conf
        })

    return results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

from torchvision import models

resnet_weights = models.ResNet50_Weights.DEFAULT

resnet_model = models.resnet50(weights=resnet_weights)

resnet_model = resnet_model.to(device)
resnet_model.eval()

print("ResNet50 loaded successfully")

# Testing

image, label, original_index = resnet_cat_dataset[0]

image_batch = image.unsqueeze(0).to(device)

result = predict(
    resnet_model,
    image_batch,
    imagenet_categories
)[0]

print("CIFAR label: cat")
print("Model prediction:", result["class_name"])
print("Confidence:", f"{result['confidence']:.4f}")

# visualise

def show_tensor_image(tensor, title="Image"):
    image = tensor.detach().cpu()

    # Undo ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image * std + mean
    image = image.clamp(0, 1)

    plt.figure(figsize=(5, 5))
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")
    plt.title(title)
    plt.show()

show_tensor_image(
    image,
    f"Original | {result['class_name']} | {result['confidence']:.2%}"
)

"""# FGSM"""

def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    images.requires_grad = True

    outputs = model(images)

    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    gradient = images.grad.detach()

    perturbation = epsilon * gradient.sign()

    adversarial_images = images + perturbation

    # ImageNet normalization parameters
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=device
    ).view(1, 3, 1, 1)

    # Valid normalized range corresponding to pixel values [0, 1]
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std

    adversarial_images = torch.max(
        torch.min(adversarial_images, upper),
        lower
    )

    return adversarial_images.detach(), perturbation.detach()

# untarget attack to move image away from its curr class

image, _, _ = resnet_cat_dataset[0]

image = image.unsqueeze(0).to(device)

# Get current model prediction
with torch.no_grad():
    output = resnet_model(image)
    original_label = output.argmax(dim=1)

print(
    "Original prediction:",
    imagenet_categories[original_label.item()]
)

import torch.nn.functional as F

epsilon = 0.02

adv_image, perturbation = fgsm_attack(
    resnet_model,
    image,
    original_label,
    epsilon
)

adv_result = predict(
    resnet_model,
    adv_image,
    imagenet_categories
)[0]

print("Original:",
      imagenet_categories[original_label.item()])

print("After FGSM:",
      adv_result["class_name"])

print("Confidence:",
      f"{adv_result['confidence']:.4f}")

"""## Visualizing all"""

def denormalize(tensor):
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=tensor.device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=tensor.device
    ).view(1, 3, 1, 1)

    return (tensor * std + mean).clamp(0, 1)


original_display = denormalize(image)[0].cpu()
adv_display = denormalize(adv_image)[0].cpu()

perturb_display = perturbation[0].abs().cpu()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_display.permute(1, 2, 0))
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(adv_display.permute(1, 2, 0))
axes[1].set_title("FGSM Adversarial")
axes[1].axis("off")

axes[2].imshow(perturb_display.permute(1, 2, 0))
axes[2].set_title("Perturbation")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("fgsm_example.png", dpi=200)
plt.show()



"""# **PGD**"""

def pgd_attack(
    model,
    images,
    labels,
    epsilon=0.02,
    alpha=0.005,
    num_steps=10,
    random_start=True
):
    """
    Projected Gradient Descent attack under an L-infinity constraint.
    """

    original_images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    if random_start:
        delta = torch.empty_like(original_images).uniform_(
            -epsilon,
            epsilon
        )

        adv_images = original_images + delta

    else:
        adv_images = original_images.clone()

    # ImageNet normalized valid range
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=device
    ).view(1, 3, 1, 1)

    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std

    for _ in range(num_steps):

        adv_images.requires_grad = True

        outputs = model(adv_images)

        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()

        loss.backward()

        gradient = adv_images.grad.detach()

        # Gradient ascent on loss
        adv_images = adv_images.detach() + alpha * gradient.sign()

        # Project back into L-infinity epsilon ball
        delta = torch.clamp(
            adv_images - original_images,
            min=-epsilon,
            max=epsilon
        )

        adv_images = original_images + delta

        # Keep valid image values
        adv_images = torch.max(
            torch.min(adv_images, upper),
            lower
        )

    return adv_images.detach()

# Applying pgd on the image

epsilon = 0.02
alpha = 0.005
steps = 10

pgd_image = pgd_attack(
    resnet_model,
    image,
    original_label,
    epsilon=epsilon,
    alpha=alpha,
    num_steps=steps,
    random_start=True
)

pgd_result = predict(
    resnet_model,
    pgd_image,
    imagenet_categories
)[0]

print("Original:",
      imagenet_categories[original_label.item()])

print("After PGD:",
      pgd_result["class_name"])

print("Confidence:",
      f"{pgd_result['confidence']:.4f}")

# visualising pgd

pgd_display = denormalize(pgd_image)[0].cpu()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_display.permute(1, 2, 0))
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(adv_display.permute(1, 2, 0))
axes[1].set_title("FGSM")
axes[1].axis("off")

axes[2].imshow(pgd_display.permute(1, 2, 0))
axes[2].set_title("PGD")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("fgsm_vs_pgd.png", dpi=200)
plt.show()

"""# **Increasing no. of images and checking accuracy**"""

# SELECTING
NUM_IMAGES = 100

experiment_indices = cat_indices[:NUM_IMAGES]

print("Images used:", len(experiment_indices))

# CREATING batch loader
resnet_experiment_dataset = CIFARCatDataset(
    raw_dataset,
    experiment_indices,
    resnet_transform
)

vit_experiment_dataset = CIFARCatDataset(
    raw_dataset,
    experiment_indices,
    vit_transform
)

resnet_loader = DataLoader(
    resnet_experiment_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2
)

vit_loader = DataLoader(
    vit_experiment_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2
)

"""# Evaluation Function"""

def evaluate_model(
    model,
    loader,
    categories,
    cat_imagenet_ids,
    epsilon=0.02,
    alpha=0.005,
    pgd_steps=10,
    max_batches=None
):

    model.eval()

    cat_imagenet_ids = set(cat_imagenet_ids)

    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0

    total = 0

    for batch_idx, (images, _, _) in enumerate(loader):

        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)

        batch_size = images.size(0)
        total += batch_size

        # -------------------------
        # Clean predictions
        # -------------------------

        with torch.no_grad():
            clean_outputs = model(images)

        clean_predictions = clean_outputs.argmax(dim=1)

        clean_correct += sum(
            pred.item() in cat_imagenet_ids
            for pred in clean_predictions
        )

        # -------------------------
        # Use clean prediction
        # as attack label
        # -------------------------

        attack_labels = clean_predictions.detach()

        # -------------------------
        # FGSM
        # -------------------------

        fgsm_images, _ = fgsm_attack(
            model,
            images,
            attack_labels,
            epsilon
        )

        with torch.no_grad():
            fgsm_outputs = model(fgsm_images)

        fgsm_predictions = fgsm_outputs.argmax(dim=1)

        fgsm_correct += sum(
            pred.item() in cat_imagenet_ids
            for pred in fgsm_predictions
        )

        # -------------------------
        # PGD
        # -------------------------

        pgd_images = pgd_attack(
            model,
            images,
            attack_labels,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=pgd_steps,
            random_start=True
        )

        with torch.no_grad():
            pgd_outputs = model(pgd_images)

        pgd_predictions = pgd_outputs.argmax(dim=1)

        pgd_correct += sum(
            pred.item() in cat_imagenet_ids
            for pred in pgd_predictions
        )

        print(
            f"Processed {total} images...",
            end="\r"
        )

    results = {
        "Clean Accuracy": clean_correct / total,
        "FGSM Accuracy": fgsm_correct / total,
        "PGD Accuracy": pgd_correct / total,
        "Images": total
    }

    return results

"""# Res net experiment"""

resnet_results = evaluate_model(
    resnet_model,
    resnet_loader,
    imagenet_categories,
    CAT_IMAGENET_IDS,
    epsilon=0.02,
    alpha=0.005,
    pgd_steps=10
)

print("\nResNet50 Results")
print("----------------")
for key, value in resnet_results.items():
    if "Accuracy" in key:
        print(key, f"{value:.2%}")
    else:
        print(key, value)

"""# ViT experiment"""

from torchvision import models

vit_weights = models.ViT_B_16_Weights.DEFAULT

vit_model = models.vit_b_16(weights=vit_weights)

vit_model = vit_model.to(device)
vit_model.eval()

print("ViT-B/16 loaded successfully")

vit_results = evaluate_model(
    vit_model,
    vit_loader,
    imagenet_categories,
    CAT_IMAGENET_IDS,
    epsilon=0.02,
    alpha=0.005,
    pgd_steps=10
)

print("\nViT-B/16 Results")
print("----------------")
for key, value in vit_results.items():
    if "Accuracy" in key:
        print(key, f"{value:.2%}")
    else:
        print(key, value)

"""# **Benchmark table**"""

import pandas as pd
comparison = pd.DataFrame({
    "Model": [
        "ResNet50",
        "ViT-B/16"
    ],
    "Clean Accuracy": [
        resnet_results["Clean Accuracy"],
        vit_results["Clean Accuracy"]
    ],
    "FGSM Accuracy": [
        resnet_results["FGSM Accuracy"],
        vit_results["FGSM Accuracy"]
    ],
    "PGD Accuracy": [
        resnet_results["PGD Accuracy"],
        vit_results["PGD Accuracy"]
    ],
    "Images": [
        resnet_results["Images"],
        vit_results["Images"]
    ]
})

comparison

# saving results

comparison.to_csv(
    "resnet_vs_vit_adversarial_results.csv",
    index=False
)

print("Saved: resnet_vs_vit_adversarial_results.csv")

"""# **Benchmark Graph**"""

plot_data = comparison.set_index("Model")[
    [
        "Clean Accuracy",
        "FGSM Accuracy",
        "PGD Accuracy"
    ]
]

plot_data.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("ResNet50 vs ViT-B/16 Under Adversarial Attacks")
plt.ylim(0, 1)

plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()

plt.savefig(
    "resnet_vs_vit_comparison.png",
    dpi=200
)

plt.show()

"""## Multiple epsilon values"""

EPSILONS = [
    0.0,
    0.005,
    0.01,
    0.02,
    0.04
]

def evaluate_attack_strengths(
    model,
    loader,
    categories,
    cat_imagenet_ids,
    epsilons,
    pgd_alpha=0.005,
    pgd_steps=10,
    max_batches=None
):

    results = []

    cat_ids = set(cat_imagenet_ids)

    for epsilon in epsilons:

        print(f"\nTesting epsilon = {epsilon}")

        clean_correct = 0
        fgsm_correct = 0
        pgd_correct = 0
        total = 0

        for batch_idx, (images, _, _) in enumerate(loader):

            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)

            total += images.size(0)

            # Clean
            with torch.no_grad():
                outputs = model(images)

            clean_pred = outputs.argmax(dim=1)

            clean_correct += sum(
                p.item() in cat_ids
                for p in clean_pred
            )

            # Attack using current prediction
            attack_labels = clean_pred.detach()

            # FGSM
            if epsilon == 0:
                fgsm_images = images.clone()
            else:
                fgsm_images, _ = fgsm_attack(
                    model,
                    images,
                    attack_labels,
                    epsilon
                )

            with torch.no_grad():
                fgsm_pred = model(
                    fgsm_images
                ).argmax(dim=1)

            fgsm_correct += sum(
                p.item() in cat_ids
                for p in fgsm_pred
            )

            # PGD
            if epsilon == 0:
                pgd_images = images.clone()
            else:
                alpha = min(pgd_alpha, epsilon)

                pgd_images = pgd_attack(
                    model,
                    images,
                    attack_labels,
                    epsilon=epsilon,
                    alpha=alpha,
                    num_steps=pgd_steps,
                    random_start=True
                )

            with torch.no_grad():
                pgd_pred = model(
                    pgd_images
                ).argmax(dim=1)

            pgd_correct += sum(
                p.item() in cat_ids
                for p in pgd_pred
            )

        results.append({
            "epsilon": epsilon,
            "clean_accuracy": clean_correct / total,
            "fgsm_accuracy": fgsm_correct / total,
            "pgd_accuracy": pgd_correct / total,
            "images": total
        })

    return pd.DataFrame(results)

"""## For ResNet"""

resnet_epsilon_results = evaluate_attack_strengths(
    resnet_model,
    resnet_loader,
    imagenet_categories,
    CAT_IMAGENET_IDS,
    EPSILONS,
    pgd_alpha=0.005,
    pgd_steps=10
)

resnet_epsilon_results

# save
resnet_epsilon_results.to_csv(
    "resnet_epsilon_results.csv",
    index=False
)

"""## For ViT"""

vit_epsilon_results = evaluate_attack_strengths(
    vit_model,
    vit_loader,
    imagenet_categories,
    CAT_IMAGENET_IDS,
    EPSILONS,
    pgd_alpha=0.005,
    pgd_steps=10
)

vit_epsilon_results

# Save
vit_epsilon_results.to_csv(
    "vit_epsilon_results.csv",
    index=False
)

"""# **Ploting attack strength**"""

# ResNet
plt.figure(figsize=(9, 6))

plt.plot(
    resnet_epsilon_results["epsilon"],
    resnet_epsilon_results["fgsm_accuracy"],
    marker="o",
    label="FGSM"
)

plt.plot(
    resnet_epsilon_results["epsilon"],
    resnet_epsilon_results["pgd_accuracy"],
    marker="o",
    label="PGD"
)

plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("ResNet50 Robustness vs Epsilon")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "resnet_robustness_vs_epsilon.png",
    dpi=200
)

plt.show()

# ViT
plt.figure(figsize=(9, 6))

plt.plot(
    vit_epsilon_results["epsilon"],
    vit_epsilon_results["fgsm_accuracy"],
    marker="o",
    label="FGSM"
)

plt.plot(
    vit_epsilon_results["epsilon"],
    vit_epsilon_results["pgd_accuracy"],
    marker="o",
    label="PGD"
)

plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("ViT-B/16 Robustness vs Epsilon")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "vit_robustness_vs_epsilon.png",
    dpi=200
)

plt.show()

IMAGE_COUNTS = [10, 50, 100, 500]