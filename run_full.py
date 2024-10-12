import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import argparse
from utils import *
from transformers import AutoTokenizer, AutoModel
from dataset import SentimentDataset
from model import fn_cls
from trainers import Finetune_Trainer
import time
timestamp = time.strftime("%m%d%H%M", time.localtime())

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='your_path')
    argparser.add_argument('--train_data', type=str, default='train_data_3_chinese')
    argparser.add_argument('--model', type=str, default='bert-base-uncased')
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--epoch', type=int, default=4)
    argparser.add_argument('--lr', type=float, default=1e-5)
    argparser.add_argument('--ckpt_name', type=str, default="3_eng")
    argparser.add_argument('--num_classes', type=int, default=2)
    


    args = argparser.parse_args()
    args.no_cuda = False

    data_path = args.data_path
    train_data = data_path + args.train_data + '.json'
    ckpt_path = 'ckpt/' + args.ckpt_name + str(timestamp) +'.pth'
    data = pd.read_json(train_data, encoding='utf-8')
    data.columns = ['text', 'label']
    text = data['text'].tolist()
    label = data['label'].tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids, attention_mask = text2token(text, tokenizer, max_length=512)
    
    data['input_ids']=input_ids
    data['attention_mask']=attention_mask

    train_data = data.sample(frac=0.8)
    test_data=data[~data.index.isin(train_data.index)]
    print(len(train_data),len(test_data))

    train_data=train_data.reset_index(drop=True)
    test_data=test_data.reset_index(drop=True)


    train_loader = DataLoader(
    SentimentDataset(train_data), 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=0
    )
    test_loader = DataLoader(
        SentimentDataset(test_data), 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )

    model = fn_cls(args, tokenizer)
    trainer = Finetune_Trainer(model, train_loader, test_loader, args)

    acc_cri = 0
    for epoch in range(args.epoch):
        train_loss = trainer.train(epoch)
        acc, _ = trainer.test(epoch)
        if acc > acc_cri:
            acc_cri = acc
            torch.save(model.state_dict(), ckpt_path)
            print("Model saved")
        else:
            break

main()










