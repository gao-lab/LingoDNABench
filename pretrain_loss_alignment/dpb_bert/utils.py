import torch
import json

def get_param_num(model):
    num_param0 = sum(p.numel() for p in model.parameters())
    num_param1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("===========================")
    print("Total params:", num_param0)
    print("Trainable params:", num_param1)
    print("Non-trainable params:", num_param0 - num_param1)
    print("===========================")

# load model checkpoint
def load_model_checkpoint(model, model_checkpoint, device):
    #loading checkpoint
    checkpoint = torch.load(model_checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    #remove unused keys
    model_state = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state}
    model_state.update(state_dict)

    #load model
    model.load_state_dict(model_state)
    return model

