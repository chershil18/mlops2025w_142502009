#!/usr/bin/env python3

import json
import toml
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ---------------------
# Image preprocessing and inference
# ---------------------
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # add batch dimension

def run_inference(model, image_path, device):
    model.eval()
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        top_probs, top_idxs = torch.topk(probs, 5)
    print("Top predictions:")
    for p, idx in zip(top_probs[0], top_idxs[0]):
        print(f"Class {idx.item()} -> Probability {p.item():.4f}")


import itertools
import torch.optim as optim

def run_grid_search(grid_json, model_cfgs, arch, device):
    arch_key = arch.lower()
    if arch_key in model_cfgs:
        base_params = model_cfgs[arch_key]
    else:
        base_params = model_cfgs.get("defaults", {})

    learning_rates = grid_json["grid"]["learning_rate"]
    optimizers = grid_json["grid"]["optimizer"]
    momentums = grid_json["grid"]["momentum"]

    combinations = list(itertools.product(learning_rates, optimizers, momentums))

    print(f"Running grid search for {arch} ({len(combinations)} combinations)...")
    results = []

    for lr, opt_name, momentum in combinations:
        print(f"\nTrying lr={lr}, optimizer={opt_name}, momentum={momentum}")
        model = get_resnet(arch, num_classes=base_params.get("num_classes", 1000), pretrained=False)
        model.to(device)

        if opt_name.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif opt_name.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            print(f"Unknown optimizer {opt_name}, skipping")
            continue

        # tiny dummy training loop
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        inputs = torch.randn(4, 3, 224, 224, device=device)
        labels = torch.randint(0, base_params.get("num_classes", 1000), (4,), device=device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        results.append({"lr": lr, "optimizer": opt_name, "momentum": momentum, "loss": loss.item()})
        print(f"Done -> loss={loss.item():.4f}")

    print("\nGrid search summary:")
    for r in results:
        print(r)

        
# ---------------------
# Load configs
# ---------------------
with open("data_and_arch.json") as f:
    config_json = json.load(f)

model_cfgs = toml.load("model_params.toml")

arch = config_json["model"]["name"]
num_classes = config_json["model"]["num_classes"]
pretrained = config_json["model"]["pretrained"]

# ---------------------
# Load model
# ---------------------
def get_resnet(arch_name, num_classes, pretrained=True):
    arch_name = arch_name.lower()
    if arch_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif arch_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif arch_name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
    elif arch_name == "resnet152":
        model = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")
    if num_classes != 1000:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet(arch, num_classes, pretrained)
model.to(device)

print(f"Loaded {arch} model on {device}")


image_path = "sample.jpg"
run_inference(model, image_path, device)


import json

# Load grid search JSON
with open("grid_search.json") as f:
    grid_json = json.load(f)


run_grid_search(grid_json, model_cfgs, arch="resnet34", device=device)
