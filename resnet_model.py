# src/model/resnet_model.py
import torch
import torchvision.models as models


def load_resnet_model(num_classes=6):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.eval()
    return model
