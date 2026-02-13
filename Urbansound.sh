#!/bin/bash
set -e

CONFIG="./configs/config.yaml"
SAVE_DIR="./checkpoints"
CSV_DIR="./csv"

mkdir -p "$SAVE_DIR"
mkdir -p "$CSV_DIR"

echo "========================================================="
echo "🚀 [Start] Full Pipeline - All Folds & All Bitrates!"
echo "========================================================="

for DB in "urbansound"; do
    ROOT="/data/ACoM/UrbanSound8K"; CSV="/data/ACoM/UrbanSound8K/metadata/UrbanSound8K.csv"
    MAX_FOLD=10

    # 폴드 1부터 MAX_FOLD까지 전부 돌립니다
    for FOLD in $(seq 1 $MAX_FOLD); do
        echo "🔥 [Dataset: $DB] Starting FOLD $FOLD / $MAX_FOLD"

        for MODEL in "beats" "ast"; do
            # 원하시는 모든 코덱과 비트레이트 조합을 다 넣었습니다!
            for CONDITION in "encodec:1.5" "encodec:3.0" "encodec:6.0" "encodec:12.0" "encodec:24.0" "opus:6.0" "opus:12.0" "opus:24.0"; do
                
                IFS=":" read -r CODEC BR <<< "$CONDITION"
                
                # 체크포인트 이름에도 fold 명시
                CKPT_NAME="${DB}_${MODEL}_fold${FOLD}_${CODEC}_${BR}k_best.pt"
                CKPT_PATH="$SAVE_DIR/$CKPT_NAME"

                if [ ! -f "$CKPT_PATH" ]; then
                    echo "▶️ [Train] $CKPT_NAME 학습 중..."
                    python train.py --config "$CONFIG" --dataset "$DB" --backbone "$MODEL" --codec "$CODEC" --bitrate "$BR" --fold "$FOLD" --data_root "$ROOT" --meta_csv "$CSV" --save_dir "$SAVE_DIR"
                fi
                
                for METHOD in "forward_only" "back_propagation"; do
                    for MODE in "freq" "time"; do
                        
                        # CSV 이름에도 fold 명시
                        CSV_NAME="eval_${DB}_${MODEL}_fold${FOLD}_${METHOD}_${CODEC}_${MODE}_${BR}k.csv"
                        CSV_PATH="$CSV_DIR/$CSV_NAME"

                        if [ ! -f "$CSV_PATH" ] && [ -f "$CKPT_PATH" ]; then
                            echo "🔎 [Eval] $CSV_NAME 분석 중..."
                            python eval.py --config "$CONFIG" --model_path "$CKPT_PATH" --dataset "$DB" --backbone "$MODEL" --codec "$CODEC" --bitrate "$BR" --fold "$FOLD" --data_root "$ROOT" --meta_csv "$CSV" --method "$METHOD" --mode "$MODE" --step 10
                        fi
                        
                    done
                done
            done
        done
    done
done