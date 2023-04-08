import json, torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

with open("encoder.json", "r") as f:
    encoder = json.load(f)

    
ocr_transformer = A.Compose([
    A.Resize(width=112, height=48),
    # A.Normalize(mean=[133.98177105, 131.71614195, 142.1675694], std=[11.89496205, 12.33475545, 11.7366861]),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0525, scale_limit=0.1, rotate_limit=1, p=1),
    A.GridDistortion(p=.2, border_mode=1),
    A.OneOf([
        A.IAASharpen(),
        A.IAAEmboss(),
        A.RandomBrightnessContrast(p=0.2),
    ]),
    ToTensorV2(),
])


val_transformer = A.Compose([
    A.Resize(width=112, height=48),
    # A.Normalize(mean=[133.98177105, 131.71614195, 142.1675694], std=[11.89496205, 12.33475545, 11.7366861]),
    # ToTensorV2(),
])


def remove_duplicates(output, decoder, blank="ε"):
    text, add = "-", 0
    for char in output:
        char = decoder[char.item()]
        if char != blank:
            if text[-1] != char:
                text += char
                add = 0
                
            else:
                if add: 
                    text += char 
                    add = 0
        else: add = 1
                    
    return text[1:]


def save_model(model_name, epoch, prev_loss, current_loss, model, optimizer):
    if prev_loss > current_loss:
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": current_loss,
        }, f"models/best_{model_name}.pt") # create models folder! 
        print("The best model was saved!")
        prev_loss = current_loss
    
    torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": current_loss,
        }, f"models/last_{model_name}.pt")
    return prev_loss




