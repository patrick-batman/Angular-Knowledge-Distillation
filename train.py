import torch
from utils import create_loader
from models import train_cosine_loss , student_model , teacher_model
import os
#                                           TRAINING OG

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--batchsize",type=int)
parser.add_argument("--epochs",type=int)
parser.add_argument("--lr",type=float)
parser.add_argument("--lmda",type=float)
parser.add_argument("--workers",type=int)
parser.add_argument("--save-dir",type=str)  
parser.add_argument("--device")

args = parser.parse_args()

# Hyperparameters
batch_size = 8
num_epochs = 15
learning_rate = 0.001
temperature = 3  # Temperature for softened distribution
lmbda = 0.5  # Lambda parameter for loss weighting
momentum = 0.9
num_workers = 0
save_dir = "output_logs"
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

if(args.save_dir):
    save_dir = args.save_dir    

if(args.device):
    if(args.device == "cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if(args.device == "mps"):
        device = "mps" if  torch.backends.mps.is_available() else "cpu"


print(args)
print("Using device: {} ".format(device))


if not os.path.exists(save_dir):
        os.makedirs(save_dir)

train_loader = create_loader(batch_size= batch_size , num_workers= num_workers)

train_cosine_loss(teacher_model,student_model,train_loader,num_epochs,learning_rate,lmbda,1-lmbda,device, log_file="loss_log.txt", model_save_dir=save_dir)
