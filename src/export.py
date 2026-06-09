import torch
import os
import sys
sys.path.insert(0, '.')
import json

if __name__ == '__main__':
    from src.dataset import download_tiny_imagenet, get_loaders
    from src.resnet18 import ResNet18
    from src.plain_cnn import PlainCNN
    from src.train import train_model
    from src.evaluate import plot_comparison

    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    download_tiny_imagenet()
    train_loader, val_loader, num_classes = get_loaders()

    classes = sorted(os.listdir('./data/tiny-imagenet-200/train'))
    with open('models/classes.json', 'w') as f:
        json.dump(classes, f)

    print("\n=== Training ResNet-18 ===")
    resnet = ResNet18(num_classes=num_classes)
    resnet_history = train_model(resnet, train_loader, val_loader, epochs=30)
    torch.save(resnet.state_dict(), 'models/resnet18.pth')

    print("\n=== Training Plain CNN baseline ===")
    plain = PlainCNN(num_classes=num_classes)
    plain_history = train_model(plain, train_loader, val_loader, epochs=30)
    torch.save(plain.state_dict(), 'models/plain_cnn.pth')

    plot_comparison(resnet_history, plain_history)

    resnet.eval()
    dummy = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        resnet, dummy, 'models/resnet18.onnx',
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11
    )
    print("Exported models/resnet18.onnx")