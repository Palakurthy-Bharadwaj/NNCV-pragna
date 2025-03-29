wandb login
python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.0015 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet-training-v2" \
    --image-height 384 \
    --image-width 768