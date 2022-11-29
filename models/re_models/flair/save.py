import torch

model = torch.load("final-model.pt")
torch.save(model, "pytorch_model.bin")
