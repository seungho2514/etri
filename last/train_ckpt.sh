#!/bin/bash

# 에러 발생 시 중단
set -e 

# 1. 기본 설정
CONFIG_BASE="./configs/config.yaml"
FOLD=1
SAVE_DIR="./checkpoints"

# PYTHONPATH 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "========================================================="
echo "🛠️  [Train Only] 체크포인트 생성 모드를 시작합니다."
echo "========================================================="

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

    # 3. 루프: 코덱별 설정 (Codec:Bitrate)
    for CODEC_INFO in "encodec:1.5" "soundstream:3.0" "opus:6.0"; do
        
        # 문자열 분리 (예: encodec:1.5 -> CODEC=encodec, BR=1.5)
        IFS=":" read -r CODEC BR <<< "$CODEC_INFO"
        
        # 4. 루프: 백본 모델
        for MODEL in "beats" "ast"; do
            
            # 파일이 이미 있는지 체크 (선택사항)
            TARGET_FILE="$SAVE_DIR/${DB}_${MODEL}_${CODEC}_best.pt"
            if [ -f "$TARGET_FILE" ]; then
                echo "⏩ [Skip] 이미 존재함: $TARGET_FILE"
                continue
            fi

            echo "---------------------------------------------------------"
            echo "▶️  Training: [$DB] | Backbone: $MODEL | Codec: $CODEC ($BR k)"
            echo "---------------------------------------------------------"

            # Python 학습 스크립트 실행
            python train_ckpt.py \
                --config "$CONFIG_BASE" \
                --dataset "$DB" \
                --backbone "$MODEL" \
                --codec "$CODEC" \
                --bitrate "$BR" \
                --fold "$FOLD" \
                --data_root "$ROOT" \
                --meta_csv "$CSV" \
                --save_dir "$SAVE_DIR"

        done
    done
done

echo ""
echo "🎉 모든 체크포인트 생성(학습)이 완료되었습니다!"