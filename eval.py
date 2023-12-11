import torch
from utils import create_loader
from models import student_model, check_accuracy


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--batchsize",type=int)
parser.add_argument("--workers",type=int)
parser.add_argument("--device")
parser.add_argument("--test-size",type=float)
parser.add_argument("--model-path",type=str)


args = parser.parse_args()

# Hyperparameters
batch_size = 8
num_workers = 0
device = "cpu"
test_size = 0.1
model_path = ""

if(args.batchsize):
    batch_size = args.batchsize

if(args.workers):
    num_workers = args.workers

if(args.device):
    if(args.device == "cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if(args.device == "mps"):
        device = "mps" if  torch.backends.mps.is_available() else "cpu"

if(args.test_size):
    test_size = args.test_size

if(args.model_path):
    model_path = args.model_path
else:
    print("Please specify model path")
    exit()


print(args)

print("Using device: {} ".format(device))

# Load Model
model = student_model
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print("Model loaded from epoch {} with loss {}".format(epoch,loss))

model = model.eval()
model.to(device)

test_loader = create_loader(batch_size= batch_size , num_workers= num_workers, fraction=test_size)
check_accuracy(model,test_loader,device)