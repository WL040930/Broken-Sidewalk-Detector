# Sidewalk Detection Using Faster R-CNN

This project aims to **detect broken sidewalks** using a custom **Faster R-CNN model** trained on images with bounding box annotations. The model is trained using PyTorch and torchvision.

---

## Requirements

Ensure the following dependencies are installed:

- Python 3.x
- PyTorch
- torchvision
- OpenCV
- pandas
- matplotlib

You can install the dependencies via `pip`:

```bash
pip install torch torchvision opencv-python pandas matplotlib
```

## Dataset 
The dataset used in this project is the Damaged Sidewalks v5 dataset from Roboflow. It contains images of sidewalks with annotations for damaged areas. The dataset is available at:

https://universe.roboflow.com/damagedsidewalks/damaged-sidewalks/dataset/5