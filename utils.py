import json, torch
from torchvision import transforms 


with open("encoder.json", "r") as f:
    encoder = json.load(f)


ocr_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.52541871, 0.51653389, 0.55751988], std=[0.04664691, 0.04837159, 0.04602622]),
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




