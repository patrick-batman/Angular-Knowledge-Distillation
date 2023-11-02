import torch
from utils import create_train_loader
from models import train_cosine_loss , student_model , teacher_model
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
batch_size = 8
num_epochs = 10
learning_rate = 0.1
temperature = 3  # Temperature for softened distribution
lmbda = 0.25
momentum = 0.9
num_workers = 0
device = "cpu"


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



train_loader = create_train_loader(batch_size= batch_size , num_workers= num_workers)
train_cosine_loss(teacher_model,student_model,train_loader,num_epochs,learning_rate,lmbda,1-lmbda,device)
