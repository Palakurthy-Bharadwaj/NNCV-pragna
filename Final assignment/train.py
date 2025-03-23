

import os
from argparse import ArgumentParser
import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose, Normalize, Resize, ToImage, ToDtype, InterpolationMode,
    RandomHorizontalFlip, ColorJitter,
)
from unet import UNet
from torch.amp import GradScaler, autocast



# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def dice_loss(pred, target, smooth=1, ignore_index=255):
    # Create a mask for valid pixels (not ignore_index)
    mask = (target != ignore_index)
    
    pred = pred.softmax(dim=1)
    
    # Only use valid indices for one-hot encoding
    target_masked = target.clone()
    target_masked[~mask] = 0  # Replace ignore_index with 0 temporarily
    
    # Create one-hot encoding
    target_one_hot = torch.zeros_like(pred)
    target_one_hot.scatter_(1, target_masked.unsqueeze(1), 1)
    
    # Zero out the background class where we had ignore pixels
    ignore_mask = (~mask).unsqueeze(1)
    target_one_hot[:, 0][ignore_mask.squeeze(1)] = 0
    
    # Calculate intersection and union only on valid pixels
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    
    # Calculate Dice score
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Average across classes and batch
    return 1 - dice.mean()

def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes")
    parser.add_argument("--batch-size", type=int, default=32)  # Reduced
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="unet-retrain-enhanced")
    return parser

def main(args):
    wandb.init(project="5lsm0-cityscapes-segmentation", name=args.experiment_id, config=vars(args))
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_image_transform = Compose([
        ToImage(), Resize((256, 512)), RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2), ToDtype(torch.float32, scale=True),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    target_transform = Compose([
        ToImage(), Resize((256, 512), interpolation=InterpolationMode.NEAREST),
        RandomHorizontalFlip(p=0.5), ToDtype(torch.long, scale=False)
    ])
    valid_image_transform = Compose([
        ToImage(), Resize((256, 512)), ToDtype(torch.float32, scale=True),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def train_transform_function(image, target):
        seed = torch.randint(0, 100000, (1,)).item()
        torch.manual_seed(seed)
        image = train_image_transform(image)
        torch.manual_seed(seed)
        target = target_transform(target)
        return image, target

    train_dataset = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=train_transform_function)
    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=lambda img, tgt: (valid_image_transform(img), target_transform(tgt)))
    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet(in_channels=3, n_classes=19).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    scaler = torch.amp.GradScaler('cuda')

    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images)
                ce_loss = criterion(outputs, labels)
                # Add this before dice_loss call
                dice = dice_loss(outputs, labels)
                loss = 0.5 * ce_loss + 0.5 * dice
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(f"Batch {i+1:04}/{len(train_dataloader):04} Loss: {loss.item():.4f}")
            wandb.log({"train_loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr'], "epoch": epoch + 1}, step=epoch * len(train_dataloader) + i)
        
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                losses.append(loss.item())
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)
                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)
                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)
                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)
                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()
                    wandb.log({"predictions": [wandb.Image(predictions_img)], "labels": [wandb.Image(labels_img)]}, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            wandb.log({"valid_loss": valid_loss}, step=(epoch + 1) * len(train_dataloader) - 1)
            scheduler.step(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(output_dir, f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth")
                torch.save(model.state_dict(), current_best_model_path)
        print(f"Validation Loss: {valid_loss:.4f}")
    
    print("Training complete!")
    torch.save(model.state_dict(), os.path.join(output_dir, f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"))
    wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)