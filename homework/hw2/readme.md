### Home work 2

Implemented pets detection and classification.

Used model: YOLOv8 nano (pretrained)\
Finetuning params:\
    - epochs: 30\
    - batch size: 8\
    - image size: 640x480\
Weights are saved in `runs/pets_yolov8n`\
GPU: Nvidia GeForce 1060 6Gb (old, but works)

Dataset info (excerpt from original dataset readme): 
> https://universe.roboflow.com/zero-okhjh/dogs-cats-and-birds-dltz4
>
> Provided by a Roboflow user\
> License: CC BY 4.0
>
> Model for detecting dogs, cats and birds. Images and labels taken from Google Open Images Dataset V7

We then take only a 100 images from each class for train dataset.
test and validation datasets are not changed.\
    To do this, we ran `dataset.py` once. (more details in the file itself)



Inference:\
    To run model in inference mode, run `infer.py`\
    Classified images with predicted classes and bounding boxes are stored in `runs/pets_infer`

