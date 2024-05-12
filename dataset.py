# src/data_processing/dataset.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 数据预处理
def create_dataloaders(data_dir, batch_size=32, num_workers=4, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, dataset.classes
