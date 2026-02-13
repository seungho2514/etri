import torch
import torch.nn as nn
import argparse
import os
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace

from src.utils import setup_paths
setup_paths() 

from src.dataset import get_dataset #
from src.models import AudioClassifier #
from src.codec import AudioCodec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["esc50", "urbansound"], required=True)
    parser.add_argument("--backbone", type=str, choices=["beats", "ast"], required=True)
    
    parser.add_argument("--codec", type=str, choices=["encodec", "soundstream", "opus"], required=True)
    parser.add_argument("--bitrate", type=float, default=6.0)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--meta_csv", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.backbone}_{args.codec}_{args.bitrate}k_analysis.pt")
    

    with open(args.config, 'r') as f:
        cfg = SimpleNamespace(**yaml.safe_load(f))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 50 if args.dataset == 'esc50' else 10 #
    duration = 5 if args.dataset == 'esc50' else 4

    data_root = args.data_root if args.data_root else cfg.data_root
    meta_csv = args.meta_csv if args.meta_csv else cfg.meta_csv
    # 1. 데이터셋 로드
    ds_train = get_dataset(SimpleNamespace(name=args.dataset, path=data_root, 
                                       meta_csv=meta_csv, fold=args.fold, 
                                       train=True, duration=duration))
    ds_test = get_dataset(SimpleNamespace(name=args.dataset, path=data_root, 
                                        meta_csv=meta_csv, fold=args.fold, 
                                        train=False, duration=duration))
    
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # 2. 모델 및 코덱 초기화
    model = AudioClassifier(args.backbone, num_classes, cfg.ckpt_path, duration).to(device)
    codec_mgr = AudioCodec(args.codec, device, bitrate=args.bitrate)

    optimizer = torch.optim.Adam(model.head.parameters(), lr=float(cfg.lr))
    criterion = nn.CrossEntropyLoss()

    # 3. 학습 루프 (Codec-Aware Training)
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            wav, label = batch['wav'].to(device), batch['label'].to(device)
            
            # [핵심] 원본 오디오를 즉시 압축 후 복원된 음으로 교체
            wav = codec_mgr.apply(wav)

            optimizer.zero_grad()
            loss = criterion(model(wav), label)
            loss.backward()
            optimizer.step()

        # 4. 검증 루프 (역시 압축된 소리로 평가)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                wav, label = batch['wav'].to(device), batch['label'].to(device)
                wav = codec_mgr.apply(wav) # 평가 시에도 동일 조건 적용
                preds = model(wav).argmax(dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)
        
        acc = correct / total
        print(f"📊 Fold {args.fold} Epoch {epoch+1}: Acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()