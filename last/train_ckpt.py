import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import yaml
import warnings
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# 프로젝트 모듈 로드
from src.utils import setup_paths
setup_paths()

from src.dataset import get_dataset
from src.models import AudioClassifier
from src.codec import AudioCodec 

def train_one_epoch(model, loader, criterion, optimizer, device, codec_mgr, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Train Ep {epoch}")
    for batch in pbar:
        wav, label = batch['wav'].to(device), batch['label'].to(device)
        
        # 코덱 적용 (Matched Training)
        if codec_mgr is not None:
            wav = codec_mgr.apply(wav)
            
        optimizer.zero_grad()
        outputs = model(wav)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100.*correct/total:.2f}%")
        
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device, codec_mgr):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            wav, label = batch['wav'].to(device), batch['label'].to(device)
            
            if codec_mgr is not None:
                wav = codec_mgr.apply(wav)

            outputs = model(wav)
            loss = criterion(outputs, label)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    # 필수 인자들
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--meta_csv", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--codec", type=str, required=True)
    parser.add_argument("--bitrate", type=float, required=True)
    
    args = parser.parse_args()

    # 설정 로드
    with open(args.config, 'r') as f:
        cfg = SimpleNamespace(**yaml.safe_load(f))
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 학습 시작: {args.dataset} | {args.backbone} | {args.codec} ({args.bitrate}k)")

    # [핵심 수정] 설정 파일에 키가 없어도 에러 안 나게 기본값 설정
    batch_size = getattr(cfg, 'batch_size', 32)
    num_workers = getattr(cfg, 'num_workers', 4)   # <- 에러 나던 부분 수정됨
    lr = float(getattr(cfg, 'lr', 1e-4))
    weight_decay = getattr(cfg, 'weight_decay', 0.05)
    epochs = getattr(cfg, 'epochs', 50)

    # 1. 코덱 매니저 초기화
    codec_mgr = AudioCodec(args.codec, device, bitrate=args.bitrate)

    # 2. 데이터셋 로드
    ds_train = get_dataset(SimpleNamespace(
        name=args.dataset, path=args.data_root, meta_csv=args.meta_csv,
        fold=args.fold, train=True, duration=5 if args.dataset=='esc50' else 4
    ))
    # 수정된 변수(num_workers) 사용
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    ds_val = get_dataset(SimpleNamespace(
        name=args.dataset, path=args.data_root, meta_csv=args.meta_csv,
        fold=args.fold, train=False, duration=5 if args.dataset=='esc50' else 4
    ))
    # 수정된 변수(num_workers) 사용
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 3. 모델 초기화
    num_classes = 50 if args.dataset == 'esc50' else 10
    pretrained_path = cfg.ckpt_path if hasattr(cfg, 'ckpt_path') else None
    
    model = AudioClassifier(args.backbone, num_classes, ckpt_path=pretrained_path, duration=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_acc = 0.0
    save_name = f"{args.dataset}_{args.backbone}_{args.codec}_best.pt"
    save_path = os.path.join(args.save_dir, save_name)

    # 4. 학습 루프
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, codec_mgr, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, codec_mgr)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   ✅ Best Saved: {save_name} (Acc: {best_acc:.2f}%)")
    
    print(f"🏁 학습 완료. 저장위치: {save_path}\n")

if __name__ == "__main__":
    main()