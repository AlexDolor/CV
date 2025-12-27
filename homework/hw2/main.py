# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# check avalibility
# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="homework/hw2/dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,   # GPU
        workers=2,
        project="runs",
        name="pets_yolov8n",
    )

if __name__ == "__main__":
    main()