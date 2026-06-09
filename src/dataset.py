import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(data_dir='./data/tiny-imagenet-200', batch_size=64):
    train_transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_transforms)
    val_data   = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),   transform=val_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, len(train_data.classes)


def download_tiny_imagenet(dest='./data'):
    import urllib.request, zipfile
    os.makedirs(dest, exist_ok=True)
    url  = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    path = os.path.join(dest, 'tiny-imagenet-200.zip')
    if not os.path.exists(os.path.join(dest, 'tiny-imagenet-200')):
        print("Downloading Tiny-ImageNet (~235MB)...")
        urllib.request.urlretrieve(url, path)
        print("Extracting...")
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(dest)
        os.remove(path)
        fix_val_structure(os.path.join(dest, 'tiny-imagenet-200'))
        print("Done.")
    else:
        print("Tiny-ImageNet already exists.")


def fix_val_structure(data_dir):
    """
    Tiny-ImageNet val folder is flat — restructure it into
    class subfolders so ImageFolder can read it.
    """
    val_dir      = os.path.join(data_dir, 'val')
    annotations  = os.path.join(val_dir, 'val_annotations.txt')
    if not os.path.exists(annotations):
        return
    img_to_class = {}
    with open(annotations) as f:
        for line in f:
            parts = line.strip().split('\t')
            img_to_class[parts[0]] = parts[1]
    images_dir = os.path.join(val_dir, 'images')
    for img, cls in img_to_class.items():
        cls_dir = os.path.join(val_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        src = os.path.join(images_dir, img)
        dst = os.path.join(cls_dir, img)
        if os.path.exists(src):
            os.rename(src, dst)
    print("Val folder restructured.")