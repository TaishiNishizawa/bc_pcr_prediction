import os
import argparse
import numpy as np
import torch
import json
import torch.optim as optim
import matplotlib.pyplot as plt
from data_utils import get_data_loader
from model import Model
from bc_pcr_prediction.code.train_test import train, test

def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--resnet_type", default=50, type=int, help="type of resnet")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")

    parser.add_argument("--decay", default=0.000005, type=float, help="decay rate")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--opt", default="adamw", type=str, help="type of optimizer")
    
    args = parser.parse_args()
    
    with open('', 'r') as f:
        ispy2_data_list = json.load(f)

    with open('', 'r') as f:
        ispy1_data_list = json.load(f)

    # Get indices for all data points
    train_val_data_indices = np.array(list(range(len(ispy2_data_list))))
    train_indices = ''
    val_indices = ''
    
    external_indices = list(range(len(ispy1_data_list)))

    if args.resnet_type == 18:
        model = Model(resnet_type=18)
    elif args.resnet_type == 34:
        model = Model(resnet_type=34)
    elif args.resnet_type == 50:
        model = Model(resnet_type=50)
    elif args.resnet_type == 101:
        model = Model(resnet_type=101)

    device = torch.device("cuda")
    model.to(device)

    checkpoint = torch.load("")
    model.load_state_dict(checkpoint["state_dict"])

    if args.opt == "adam":
        optimizer = optim.Adam(params = model.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(params = model.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = optim.SGD(params = model.parameters(), lr = args.lr, momentum = args.momentum, nesterov = True)
   
    train_loader  = get_data_loader("train", train_indices, data_list = ispy2_data_list, batch_size=args.batch_size)
    internal_test_loader  = get_data_loader("test", val_indices, data_list = ispy2_data_list, batch_size=1)
    external_test_loader = get_data_loader("test", external_indices, data_list = ispy1_data_list, batch_size=1)

    print(f"Training with resnet {args.resnet_type}, for {args.epochs} epochs with lr {args.lr} and optimizer {args.opt} w/ batch {args.batch_size}")
    train(model, optimizer, train_loader, internal_test_loader, external_test_loader, epochs = args.epochs)
    test(model, external_test_loader)

if __name__ == "__main__":
    main()