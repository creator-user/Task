# src/model/inference.py
import torch
from torchvision import transforms
from PIL import Image
from src.model.resnet_model import load_resnet_model
from src.config import MODEL_PATH

# TrashNet 数据集的 6 个类别标签
LABELS = [
    "Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return preprocess(image).unsqueeze(0)


def load_trained_model():
    model = load_resnet_model(num_classes=len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict(image_tensor, model):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()


def predict_label(image_tensor, model):
    """返回预测结果的类别标签和置信度"""
    index, confidence = predict(image_tensor, model)
    return LABELS[index] if 0 <= index < len(LABELS) else "Unknown", confidence
