import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Step 1: Load the model architecture
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/rcnn.pth", map_location=torch.device('cpu')))
model.eval()

# Step 2: Load and transform the test image
image_path = "Data/test/1927_tile1_png.rf.495eb65046bad59edcd8c7de17baa674.jpg" # example image path
image = Image.open(image_path).convert("RGB")
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image)

# Step 3: Add batch dimension
input_batch = [image_tensor]

# Step 4: Run inference
with torch.no_grad():
    outputs = model(input_batch)

# Step 5: Visualize results
fig, ax = plt.subplots(1)
ax.imshow(image)

for box, score in zip(outputs[0]["boxes"], outputs[0]["scores"]):
    if score > 0.5:  # Confidence threshold
        x1, y1, x2, y2 = box.numpy()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.title("Detected: Broken Sidewalk")
plt.axis('off')
plt.show()
