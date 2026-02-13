#!/bin/bash
set -e  # 에러 발생 시 즉시 중단 (안전장치)

# =======================================================
# 1. 환경 설정 및 디렉토리 생성
# =======================================================
CONFIG="./configs/config.yaml"
SAVE_DIR="./checkpoints"
CSV_DIR="./csv"  # eval.py에서 저장하는 폴더명
FOLD=1

# 폴더가 없으면 미리 생성
mkdir -p "$SAVE_DIR"
mkdir -p "$CSV_DIR"

echo "========================================================="
echo "🚀 [Start] Full Pipeline (Train -> Eval) with Smart Skip"
echo "========================================================="

# =======================================================
# 2. 실험 루프 시작
# =======================================================

# (1) 데이터셋 루프
for DB in "esc50"; do
    
    # 데이터셋 경로 자동 설정
    if [ "$DB" == "esc50" ]; then
        ROOT="/data/ACoM/ESC-50"; CSV="/data/ACoM/ESC-50/meta/esc50.csv"
    else
        ROOT="/data/ACoM/UrbanSound8K"; CSV="/data/ACoM/UrbanSound8K/metadata/UrbanSound8K.csv"
    fi

    # (2) 백본 모델 루프
    for MODEL in "beats"; do

        # (3) 코덱 및 비트레이트 루프 (형식: "코덱:비트레이트")
        for CONDITION in "encodec:1.5"; do
            
            IFS=":" read -r CODEC BR <<< "$CONDITION"
            
            # ------------------------------------------------------------
            # A. 학습 (Train) 단계 - 파일 체크 로직
            # ------------------------------------------------------------
            # [중요] Python 코드에서 저장하는 이름과 100% 일치해야 함
            # 예: esc50_beats_encodec_1.5_best.pt (비트레이트 포함 권장)
            # 만약 Python 코드에서 BR을 파일명에 안 넣었다면 수정 필요!
            CKPT_NAME="${DB}_${MODEL}_${CODEC}_${BR}k_analysis.pt"
            CKPT_PATH="$SAVE_DIR/$CKPT_NAME"

            echo ""
            echo "---------------------------------------------------------"
            echo "🏗️  [Target] $CKPT_NAME"

            if [ -f "$CKPT_PATH" ]; then
                echo "⏩ [Train Skip] 체크포인트가 이미 존재합니다."
            else
                echo "▶️  [Train Run] 학습을 시작합니다..."
                python train.py \
                    --config "$CONFIG" --dataset "$DB" --backbone "$MODEL" \
                    --codec "$CODEC" --bitrate "$BR" --fold "$FOLD" \
                    --data_root "$ROOT" --meta_csv "$CSV" --save_dir "$SAVE_DIR"
            fi

            # ------------------------------------------------------------
            # B. 평가 (Eval) 단계 - 파일 체크 로직
            # ------------------------------------------------------------
            
            # (4) Saliency Method 루프
            for METHOD in "forward_only" "back_propagation"; do
                
                # (5) Pruning Mode 루프
                for MODE in "freq" "time"; do
                    
                    # [중요] eval.py의 save_fn 변수와 100% 일치해야 함
                    # format: eval_{dataset}_{backbone}_{method}_{codec}_{mode}_{bitrate}k.csv
                    CSV_NAME="eval_${DB}_${MODEL}_${METHOD}_${CODEC}_${MODE}_${BR}k_analysis.csv"
                    CSV_PATH="$CSV_DIR/$CSV_NAME"
                    
                    # 체크포인트가 없으면 평가는 무조건 실패하므로 체크
                    if [ ! -f "$CKPT_PATH" ]; then
                        echo "⚠️  [Eval Error] 학습 모델($CKPT_NAME)이 없어서 평가를 건너뜁니다."
                        continue
                    fi

                    if [ -f "$CSV_PATH" ]; then
                        echo "   ⏩ [Eval Skip] 결과 CSV가 이미 존재함: $CSV_NAME"
                    else
                        echo "   🔎 [Eval Run] $METHOD | $MODE 분석 시작..."
                        python eval.py \
                            --config "$CONFIG" \
                            --model_path "$CKPT_PATH" \
                            --dataset "$DB" --backbone "$MODEL" \
                            --codec "$CODEC" --bitrate "$BR" --fold "$FOLD" \
                            --data_root "$ROOT" --meta_csv "$CSV" \
                            --method "$METHOD" --mode "$MODE" --step 10
                    fi
                    
                done # Mode
            done # Method

        done # Codec/Bitrate
    done # Backbone
done # Dataset

echo ""
echo "🎉 모든 작업이 완료되었습니다!"