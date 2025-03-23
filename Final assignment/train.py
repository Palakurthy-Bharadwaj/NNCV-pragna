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
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="unet-simple")
    return parser

def main(args):
    # Initialize wandb for experiment tracking
    wandb.init(project="cityscapes-segmentation", name=args.experiment_id, config=vars(args))

    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    train_image_transform = Compose([
        ToImage(), 
        Resize((256, 512)), 
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        ToDtype(torch.float32, scale=True),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet stats
    ])
    
    target_transform = Compose([
        ToImage(), 
        Resize((256, 512), interpolation=InterpolationMode.NEAREST),
        RandomHorizontalFlip(p=0.5), 
        ToDtype(torch.long, scale=False)
    ])
    
    valid_image_transform = Compose([
        ToImage(), 
        Resize((256, 512)), 
        ToDtype(torch.float32, scale=True),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Ensure synchronized random transformations for image and target
    def train_transform_function(image, target):
        seed = torch.randint(0, 100000, (1,)).item()
        torch.manual_seed(seed)
        image = train_image_transform(image)
        torch.manual_seed(seed)
        target = target_transform(target)
        return image, convert_to_train_id(target)  # Convert to train_id here

    def val_transform_function(image, target):
        image = valid_image_transform(image)
        target = target_transform(target)
        return image, convert_to_train_id(target)  # Convert to train_id here

    # Datasets and dataloaders
    train_dataset = Cityscapes(
        args.data_dir, split="train", mode="fine", 
        target_type="semantic", transforms=train_transform_function
    )
    valid_dataset = Cityscapes(
        args.data_dir, split="val", mode="fine", 
        target_type="semantic", transforms=val_transform_function
    )
    
    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Model, loss, optimizer, and scheduler
    model = UNet(in_channels=3, n_classes=19).to(device)
    
    # Using just cross entropy loss with label smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.1)
    
    # AdamW optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    best_valid_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_model.pth")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(train_dataloader)} Loss: {loss.item():.4f}")
            
            wandb.log({
                "train_loss": loss.item(), 
                "learning_rate": optimizer.param_groups[0]['lr'], 
                "epoch": epoch + 1
            })
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"  Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_dataloader):
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                
                # Log sample predictions
                if i == 0:
                    predictions = outputs.argmax(1, keepdim=True)
                    colored_predictions = convert_train_id_to_color(predictions)
                    colored_labels = convert_train_id_to_color(labels.unsqueeze(1))
                    
                    predictions_img = make_grid(colored_predictions.cpu(), nrow=4)
                    labels_img = make_grid(colored_labels.cpu(), nrow=4)
                    
                    wandb.log({
                        "predictions": wandb.Image(predictions_img.permute(1, 2, 0).numpy()),
                        "ground_truth": wandb.Image(labels_img.permute(1, 2, 0).numpy()),
                    })
            
            avg_valid_loss = valid_loss / len(valid_dataloader)
            print(f"  Validation Loss: {avg_valid_loss:.4f}")
            
            wandb.log({"valid_loss": avg_valid_loss, "epoch": epoch + 1})
            
            # Update learning rate
            scheduler.step(avg_valid_loss)
            
            # Save best model
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"  Saved new best model with validation loss: {best_valid_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, f"final_model_e{args.epochs}_loss{avg_valid_loss:.4f}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")
    
    wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)