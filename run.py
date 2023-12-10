import torch
from utils import create_loader
from models import train_cosine_loss , student_model , teacher_model, test_model
# from models import student_model , teacher_model

# print(f"PyTorch version: {torch.__version__}")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

#                                           TRAINING OG

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--batchsize",type=int)
parser.add_argument("--epochs",type=int)
parser.add_argument("--lr",type=float)
parser.add_argument("--lmda",type=float)
parser.add_argument("--workers",type=int)
parser.add_argument("--device")

args = parser.parse_args()

# Hyperparameters
batch_size = 512
num_epochs = 20
learning_rate = 0.001
temperature = 3  # Temperature for softened distribution
lmbda = 0.5  # Lambda parameter for loss weighting
momentum = 0.9
num_workers = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device: {} ".format(device))


if(args.batchsize):
    batch_size = args.batchsize

if(args.epochs):
    num_epochs = args.epochs

if(args.lr):
    learning_rate = args.lr

if(args.lmda):
    lmbda = args.lmda

if(args.workers):
    num_workers = args.workers

if(args.device):
    if(args.device == "cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if(args.device == "mps"):
        device = "mps" if  torch.backends.mps.is_available() else "cpu"
    

print(args)



train_loader, test_loader = create_loader(batch_size= batch_size , num_workers= num_workers)

train_cosine_loss(teacher_model,student_model,train_loader,num_epochs,learning_rate,lmbda,1-lmbda,device, log_file="loss_log.txt")

test_model(student_model,test_loader,device)