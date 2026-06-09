import torch
import matplotlib.pyplot as plt
import os


def plot_comparison(resnet_history, plain_history,
                    save_dir='./assets'):
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(resnet_history['train_loss'], label='ResNet-18', color='#2563eb')
    ax1.plot(plain_history['train_loss'],  label='Plain CNN', color='#dc2626',
             linestyle='--')
    ax1.set_title('Training Loss — ResNet vs Plain CNN')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(alpha=0.3)

    # Top-1 accuracy
    ax2.plot(resnet_history['val_acc'], label='ResNet-18 Top-1', color='#2563eb')
    ax2.plot(plain_history['val_acc'],  label='Plain CNN Top-1', color='#dc2626',
             linestyle='--')
    ax2.plot(resnet_history['top5_acc'], label='ResNet-18 Top-5',
             color='#16a34a')
    ax2.set_title('Validation Accuracy — ResNet vs Plain CNN')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=150)
    print("Saved assets/comparison.png")
    plt.show()


def evaluate_model(model, val_loader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    top1, top5, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            top1 += (out.argmax(1) == labels).sum().item()
            top5 += sum(
                labels[i].item() in out[i].topk(5).indices.tolist()
                for i in range(labels.size(0))
            )
            total += labels.size(0)
    print(f"Top-1: {top1/total*100:.2f}% | Top-5: {top5/total*100:.2f}%")
    return top1/total, top5/total