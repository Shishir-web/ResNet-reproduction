import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def train_model(model, train_loader, val_loader,
                epochs=30, lr=0.1, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr,
                    momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_acc': [], 'top5_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for imgs, labels in tqdm(train_loader,
                                 desc=f"Epoch {epoch+1}/{epochs}",
                                 leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        top1, top5, total = 0, 0, 0
        model.eval()
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

        t1  = top1 / total
        t5  = top5 / total
        avg = running_loss / len(train_loader)
        history['train_loss'].append(avg)
        history['val_acc'].append(t1)
        history['top5_acc'].append(t5)
        scheduler.step()
        print(f"Epoch {epoch+1:02d} | Loss: {avg:.4f} | "
              f"Top-1: {t1:.4f} | Top-5: {t5:.4f}")

    return history