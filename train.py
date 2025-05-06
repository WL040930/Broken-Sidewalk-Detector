import os
import pandas as pd
import torch
import torchvision
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR

# Custom Dataset Class for Sidewalk Damage Detection
class SidewalkDataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transforms
        self.imgs = self.annotations['filename'].unique()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_filename = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Initialize empty lists for bounding boxes and labels
        boxes = []
        labels = []

        # Get annotations for the current image
        rows = self.annotations[self.annotations['filename'] == img_filename]
        for _, row in rows.iterrows():
            boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            labels.append(1)  # Only one class: 'Losa-Agrietada'

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Additional metadata for the target
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        # Apply transformations if any
        if self.transforms:
            img = self.transforms(img)

        return img, target


# Function to initialize the model with improved backbone (ResNet101 for better performance)
def initialize_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Modify the classifier to fit our dataset (2 classes: background + sidewalk damage)
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, 2
    )
    return model


# Function to train the model with data augmentation and learning rate scheduler
def train_model(model, data_loader, optimizer, num_epochs, device):
    model.to(device)
    model.train()

    # Learning rate scheduler (decays the learning rate every 3 epochs)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # Step the scheduler
        scheduler.step()

        # Print the loss after each epoch
        print(f"Epoch {epoch + 1}, Loss: {losses.item():.4f}")


# Main function to initialize and train the model
def main():
    # Data augmentation for training
    transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),  # Horizontal flip for augmentation
        T.RandomRotation(10),  # Random rotation for robustness
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)  # Color jitter
    ])

    # Dataset and DataLoader
    dataset = SidewalkDataset(csv_file="Data/train/_annotations.csv", img_dir="Data/train/", transforms=transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Initialize model and optimizer
    model = initialize_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Determine device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    num_epochs = 5
    train_model(model, data_loader, optimizer, num_epochs, device)

    # Save the trained model
    torch.save(model.state_dict(), "models/rcnn.pth")
    print("Model saved to 'models/rcnn.pth'")

if __name__ == "__main__":
    main()
