import torch
import torch.nn.functional as F
import argparse
import os
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader
import torchaudio

# 프로젝트 경로 및 모듈 로드
from src.utils import setup_paths
setup_paths()

from src.dataset import get_dataset #
from src.models import AudioClassifier #
from src.codec import AudioCodec #

# ---------------------------------------------------------
# 1. Saliency 추출기 (트랜스포머 이전 레이어 타겟)
# ---------------------------------------------------------
class SaliencyFilter:
    def __init__(self, model, backbone_name):
        self.model = model
        self.backbone_name = backbone_name
        
        # [요청사항] 트랜스포머 통과 전 레이어 설정
        if backbone_name == 'beats':
            # BEATs: Patch Embedding 직후의 LayerNorm
            self.target_layer = model.backbone.layer_norm
        else:
            # AST: Patch Embedding 출력 레이어
            self.target_layer = model.backbone.v.patch_embed

    def get_map(self, wav, label, method="forward_only"):
        feat = None
        def hook_fn(m, i, o):
            nonlocal feat
            feat = o
            if method == "back_propagation":
                feat.retain_grad()

        handle = self.target_layer.register_forward_hook(hook_fn)
        
        if method == "forward_only":
            # 1. Forward Only: 활성화값의 크기(Magnitude)만 사용
            with torch.no_grad():
                _ = self.model(wav)
            score = feat.abs().mean(dim=-1).detach()
        else:
            # 2. Back Propagation: 정답 클래스에 대한 기울기(Gradient) 사용
            with torch.enable_grad():
                wav = wav.clone().requires_grad_(True)
                logits = self.model(wav)
                loss = F.cross_entropy(logits, label)
                self.model.zero_grad()
                loss.backward()
                # Input * Gradient 방식으로 중요도 산출
                score = torch.relu(-feat.detach() * feat.grad).mean(dim=-1)
        
        handle.remove()
        return score.squeeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["esc50", "urbansound"], required=True)
    parser.add_argument("--backbone", type=str, choices=["beats", "ast"], required=True)
    parser.add_argument("--codec", type=str, choices=["encodec", "soundstream", "opus"], required=True)
    parser.add_argument("--method", type=str, choices=["forward_only", "back_propagation"], required=True)
    parser.add_argument("--mode", type=str, default="freq", choices=["freq", "time", "pixel"])
    parser.add_argument("--bitrate", type=float, default=1.5)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--fold", type=int, default=1) 
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--meta_csv", type=str, default=None)
    args = parser.parse_args()

    # 설정 로드
    with open(args.config, 'r') as f:
        cfg = SimpleNamespace(**yaml.safe_load(f))
    data_root = args.data_root if args.data_root else cfg.data_root
    meta_csv = args.meta_csv if args.meta_csv else cfg.meta_csv
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 50 if args.dataset == 'esc50' else 10
    duration = 5 if args.dataset == 'esc50' else 4

    # 모델 초기화
    model = AudioClassifier(args.backbone, num_classes, cfg.ckpt_path, duration).to(device)

    if not os.path.exists(args.model_path):
        print(f"⚠️ [Error] 모델 파일이 없습니다: {args.model_path}")
        return

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    codec_mgr = AudioCodec(args.codec, device, bitrate=args.bitrate)
    sal_filter = SaliencyFilter(model, args.backbone)

    # 데이터셋 로드
    ds_test = get_dataset(SimpleNamespace(
        name=args.dataset, 
        path=data_root, 
        meta_csv=meta_csv, 
        fold=args.fold, 
        train=False, 
        duration=duration
    ))
    loader = DataLoader(ds_test, batch_size=1, shuffle=False)

    spec_fn = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None, return_complex=True).to(device)
    ispec_fn = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256).to(device)

    all_logs = []
    
    # ---------------------------------------------------------
    # [수정] ./csv 폴더 생성 및 저장 경로 설정
    # ---------------------------------------------------------
    save_dir = "./csv"
    os.makedirs(save_dir, exist_ok=True)  # 폴더가 없으면 생성, 있으면 넘어감
    
    file_name = f"eval_{args.dataset}_{args.backbone}_{args.method}_{args.codec}_{args.mode}_{args.bitrate}k.csv"
    save_fn = os.path.join(save_dir, file_name)  # ./csv/파일명.csv
    
    # [수정] 이미 파일이 있으면 스킵
    if os.path.exists(save_fn):
        print(f"⏩ [Skip] 결과 파일이 이미 존재합니다: {save_fn}")
        return
    
    print(f"🔍 시작: {args.dataset} | {args.backbone} | {args.method}")

    # 평가 루프
    for idx, batch in enumerate(tqdm(loader, desc=f"Eval [{args.method}]")):
        wav, label = batch['wav'].to(device), batch['label'].to(device)
        
        # 파일명 추출 (안전하게)
        if 'filename' in batch:
            fname = os.path.basename(batch['filename'][0])
        elif 'path' in batch:
            fname = os.path.basename(batch['path'][0])
        elif 'file_path' in batch:
            fname = os.path.basename(batch['file_path'][0])
        else:
            fname = f"file_{idx:05d}"

        S_orig = spec_fn(wav)
        
        # Saliency Map
        score = sal_filter.get_map(wav, label, method=args.method)
        f_grid = 8 if args.backbone == 'beats' else 12 
        t_grid = score.numel() // f_grid
        sal_2d = score[:f_grid*t_grid].view(f_grid, t_grid).unsqueeze(0).unsqueeze(0)
        sal_map = F.interpolate(sal_2d, size=(S_orig.shape[2], S_orig.shape[3]), mode='bilinear').squeeze()

        # Pruning 타겟 설정
        if args.mode == "freq":
            p_scores = sal_map.mean(dim=1); max_k = int(S_orig.shape[2] * 0.5)
        elif args.mode == "time":
            p_scores = sal_map.mean(dim=0); max_k = int(S_orig.shape[3] * 0.5)
        else: 
            p_scores = sal_map.flatten(); max_k = int(p_scores.numel() * 0.2)
            
        sorted_idx = torch.argsort(p_scores, descending=False)

        # Step별 평가
        for k in range(0, (max_k + 1)//2, args.step):
            mask = torch.ones_like(S_orig)
            if k > 0:
                low_idx = sorted_idx[:k]
                if args.mode == "freq": mask[:, :, low_idx, :] = 0
                elif args.mode == "time": mask[:, :, :, low_idx] = 0
                else: 
                    f_i = low_idx // S_orig.shape[3]; t_i = low_idx % S_orig.shape[3]
                    mask[0, 0, f_i, t_i] = 0
            
            wav_recon = ispec_fn(S_orig * mask, length=wav.shape[-1])
            wav_codec = codec_mgr.apply(wav_recon) 
            
            with torch.no_grad():
                out = model(wav_codec)
                pred = out.argmax(dim=1).item()
                prob = F.softmax(out, dim=1)[0, label.item()].item()

            all_logs.append({
                "filename": fname, "k": k, "rate": round(k/len(p_scores), 4),
                "is_correct": int(pred == label.item()), "prob": round(prob, 4),
                "method": args.method, "mode": args.mode, "codec": args.codec
            })

    # [수정] ./csv 폴더 안에 저장
    pd.DataFrame(all_logs).to_csv(save_fn, index=False)
    print(f"✅ 분석 완료 및 저장됨: {save_fn}")

if __name__ == "__main__":
    main()