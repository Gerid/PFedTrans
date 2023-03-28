#!/bin/bash
nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m dnn -algo PerAvg -gr 2000 -did 0 -bt 0.001 -go dnn -wandb True > cifar100_peravg.out 2>&1 &

nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m dnn -algo FedFomo -gr 2000 -M 5 -did 0 -go dnn -wandb True > cifar100_fedfomo.out 2>&1 &

nohup python -u main.py -lbs 4 -nc 20 -jr 1 -nb 100 -data Cifar100 -m dnn -algo FedMTL -gr 2000 -itk 4000 -did 0 -go dnn -wandb True > cifar100_FedMTL.out 2>&1 &