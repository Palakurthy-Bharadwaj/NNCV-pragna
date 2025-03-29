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
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
)
import math
from unet import UNet

def calculate_dice(pred, target, n_classes=19, ignore_index=255, smooth=1e-5):
    """
    Calculate mean Dice coefficient for semantic segmentation.
    
    Args:
        pred (torch.Tensor): Prediction tensor of shape [B, C, H, W] or [B, H, W]
        target (torch.Tensor): Target tensor of shape [B, H, W]
        n_classes (int): Number of classes
        ignore_index (int): Index to ignore in the calculation
        smooth (float): Small constant to avoid division by zero
        
    Returns:
        float: Mean Dice coefficient across all classes (excluding ignored classes)
        list: Dice coefficient for each class
    """
    if pred.dim() == 4:  # [B, C, H, W] -> [B, H, W]
        pred = pred.argmax(dim=1)
    
    dice_scores = []
    
    # Process each class
    for cls in range(n_classes):
        # Create binary masks
        pred_mask = (pred == cls).float()
        target_mask = (target == cls).float()
        
        # Mask out ignore index
        valid_mask = (target != ignore_index).float()
        pred_mask = pred_mask * valid_mask
        target_mask = target_mask * valid_mask
        
        # Calculate Dice
        intersection = (pred_mask * target_mask).sum()
        total = pred_mask.sum() + target_mask.sum()
        
        if total < 1e-8:  # If the class is not present in this batch
            dice_scores.append(float('nan'))
        else:
            dice = (2 * intersection + smooth) / (total + smooth)
            dice_scores.append(dice.item())
    
    # Calculate mean Dice (ignoring NaN values)
    valid_dices = [dice for dice in dice_scores if not math.isnan(dice)]
    mean_dice = sum(valid_dices) / len(valid_dices) if valid_dices else 0
    
    print("dice_scores :", dice_scores)
    print("mean_dice :", mean_dice)
    return mean_dice, dice_scores

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


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--image-height", type=int, default=256, help="Height for resizing images")
    parser.add_argument("--image-width", type=int, default=512, help="Width for resizing images")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    image_transform = Compose([
        ToImage(),
        Resize((args.image_height, args.image_width)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    target_transform = Compose([
        ToImage(),
        Resize((args.image_height, args.image_width), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.long, scale=False),
    ])

    def transform_function(image, target):
        return image_transform(image), target_transform(target)
                                                        
    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform_function
    )
    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform_function
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = UNet(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_valid_loss = float('inf')
    best_valid_dice = 0  # Start from 0

    best_loss_model_path = None
    best_dice_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            all_dices = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                predictions = outputs.softmax(1).argmax(1)  # Get class predictions

                # Calculate Dice for this batch
                batch_dice, class_dices = calculate_dice(predictions, labels)
                all_dices.append(batch_dice)
            
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

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            valid_dice = sum(all_dices) / len(all_dices)

            # Log metrics
            wandb.log({
                "valid_loss": valid_loss,
                "valid_dice": valid_dice
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            print(f"Validation Loss: {valid_loss:.4f}, Dice: {valid_dice:.4f}")

            # Save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"GREAT_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)
                print(f"New best model saved with validation loss: {valid_loss:.4f}")
            
            if valid_dice > best_valid_dice:
                best_valid_dice = valid_dice
                if best_dice_model_path:
                    os.remove(best_dice_model_path)
                best_dice_model_path = os.path.join(
                    output_dir, 
                    f"GREAT_dice_model-epoch={epoch:04}-dice={valid_dice:.4f}.pth"
                )
                torch.save(model.state_dict(), best_dice_model_path)
                print(f"New best Dice model saved with Dice: {valid_dice:.4f}")

        print("Training complete!")

    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_GREAT_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)