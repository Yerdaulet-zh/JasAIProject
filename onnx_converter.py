import torch
from model import Architecture


model = Architecture(num_classes=37).cuda()
model.load_state_dict(torch.load("best_models/DeepBestModel.pt")['model'])

# set the model to inference mode
model.eval()

# Input to the model
x = torch.randn(1, 3, 50, 224, requires_grad=True, device="cuda")

# Export the model
torch.onnx.export(model,                               # model being run
                  x,                                   # model input (or a tuple for multiple inputs)
                  "onnx_models/DeepConvModel.onnx",    # where to save the model (can be a file or file-like object)
                  export_params=True,                  # store the trained parameter weights inside the model file
                  do_constant_folding=True,            # whether to execute constant folding for optimization
                  input_names = ['input'],             # the model's input names
                  output_names = ['output'],           # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

