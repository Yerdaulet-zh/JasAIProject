import albumentations as A
from albumentations.pytorch import ToTensorV2


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