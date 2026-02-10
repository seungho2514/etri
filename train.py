import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import setup_paths
setup_paths()

from src.dataset import get_dataset
from src.models import AudioClassifier

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.fold is not None:
        cfg.fold = args.fold
        
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"----Training Start: {args.dataset} | Fold {args.fold} | Backbone {args.backbone}----")

    ds_cfg_train = SimpleNamespace(
        name=cfg.dataset, path=cfg.data_root, meta_csv=cfg.meta_csv,
        fold=cfg.fold, train=True, duration=duration
    )
    
    ds_cfg_test = SimpleNamespace(
        name=cfg.dataset, path=cfg.data_root, meta_csv=cfg.meta_csv,
        fold=cfg.fold, train=False, duration=duration
    )

    train_ds = get_dataset(ds_cfg_train)
    test_ds = get_dataset(ds_cfg_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # 6. Model (args -> cfg)
    model = AudioClassifier(
        backbone_name=cfg.backbone, 
        num_classes=num_classes, 
        ckpt_path=cfg.ckpt_path,
        duration=duration
    ).to(device)

    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.head.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(cfg.save_dir, exist_ok=True)
    best_acc = 0.0

    # 7. Training Loop
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            wav = batch['wav'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(wav)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                wav = batch['wav'].to(device)
                label = batch['label'].to(device)

                outputs = model(wav)
                preds = outputs.argmax(dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)
        
        test_acc = correct / total
        print(f"Epoch {epoch+1}: Loss {train_loss/len(train_loader):.4f} | Test Acc {test_acc:.4f}")

        # Save Best
        if test_acc > best_acc:
            best_acc = test_acc
            # 파일명에 fold 정보 포함
            save_name = f"{cfg.dataset}_{cfg.backbone}_fold{cfg.fold}_best.pt"
            save_path = os.path.join(cfg.save_dir, save_name)
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved Best Model: {save_path}")

if __name__ == "__main__":
    main()