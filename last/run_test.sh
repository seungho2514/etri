#!/bin/bash

# 1. 공통 설정 (콜론 : 대신 등호 = 사용, 공백 주의)
CONFIG_BASE="/workspace/etri/configs/config.yaml"
FOLD=1

# 2. 루프: 데이터셋 (ESC-50, UrbanSound8K)
for DB in "esc50" "urbansound"; do

    # 데이터셋별 경로 설정
    if [ "$DB" == "esc50" ]; then
        ROOT="/data/ACoM/ESC-50"
        CSV="/data/ACoM/ESC-50/meta/esc50.csv"
    else
        ROOT="/data/ACoM/UrbanSound8K"
        CSV="/data/ACoM/UrbanSound8K/metadata/UrbanSound8K.csv"
    fi

    # 3. 루프: 코덱별 설정
    for CODEC_INFO in "encodec:1.5" "soundstream:3.0" "opus:6.0"; do
        # 문자열 분리
        IFS=":" read -r CODEC BR <<< "$CODEC_INFO"
        
        # 4. 루프: 백본 모델
        for MODEL in "beats" "ast"; do
            
            echo "================================================================="
            echo "🔥 실험 시작: [$DB] | 모델: $MODEL | 코덱: $CODEC ($BR kbps)"
            echo "================================================================="

            # [A] 학습 (Train)
            python train.py \
                --config "$CONFIG_BASE" \
                --dataset "$DB" \
                --backbone "$MODEL" \
                --codec "$CODEC" \
                --bitrate "$BR" \
                --fold "$FOLD" \
                --data_root "$ROOT" \
                --meta_csv "$CSV"

            # [B] 학습된 모델 체크포인트 경로
            MODEL_FILE="./checkpoints/${DB}_${MODEL}_${CODEC}_best.pt"

            # [C] 평가 (Eval)
            for METHOD in "forward_only" "back_propagation"; do
                echo "🔍 평가 중: $METHOD 방식..."
                python eval.py \
                    --config "$CONFIG_BASE" \
                    --model_path "$MODEL_FILE" \
                    --dataset "$DB" \
                    --backbone "$MODEL" \
                    --codec "$CODEC" \
                    --method "$METHOD" \
                    --bitrate "$BR" \
                    --mode freq \
                    --data_root "$ROOT" \
                    --meta_csv "$CSV"
            done
            
            echo "✅ [$DB | $MODEL | $CODEC] 조합 완료!"
            echo ""
        done
    done
done

echo "🏁 모든 12개 실험이 종료되었습니다."