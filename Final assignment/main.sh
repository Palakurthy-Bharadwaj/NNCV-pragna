wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 60 \
    --lr 0.002 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet-training" \