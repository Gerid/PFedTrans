#!/bin/bash

nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m dnn -algo FedAvg -gr 1000 -did 0 -go dnn -wandb True > cifar100_fedavg.out 2>&1 &

nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m dnn -algo FedProx -gr 1000 -did 0 -mu 0.001 -go dnn -wandb True > cifar100_fedprox.out 2>&1 &

nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m dnn -algo pFedMe -gr 1000 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -go dnn -wandb True > cifar100_pfedme.out 2>&1 &