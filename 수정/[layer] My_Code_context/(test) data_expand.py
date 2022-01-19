import pandas as pd
import numpy as np
import os
import csv
import re
import string
from pathlib import Path

# index 추가 전의 데이터를 불러옴.
train = pd.read_csv('/home/ojoo/SemEval-2022-Task2/MelBERT/My_Code_context copy/data/preproc/train.csv', encoding='utf-8', sep=',')
len(train)
train.isna().sum()
train = train.fillna('')
train.isna().sum()







new_list = []

for i in range(len(train)):
        
    ex = train.iloc[i,:]

    id = ex['ID'] # str
    label = ex['label'] # int
    target = ex['target'] # str
    prev = ex['previous'] # str
    next = ex['next'] # str
    sent = ex['sentence'] # str
    mwe = ex['mwe'] # str

    new_id = id + '-1'

    if label==0 and (mwe in prev):

        new_label = 0
        new_target = prev
        new_prev = ""
        new_next = target
        new_sent = new_prev + new_target + new_next
        new_mwe = mwe
        new_list.append([new_id, new_label, new_target, new_prev, new_next, new_sent, new_mwe])


    if label==0 and (mwe in next):
        
        new_label = 0
        new_target = next
        new_prev = target
        new_next = ""
        new_sent = new_prev + new_target + new_next
        new_mwe = mwe
        new_list.append([new_id, new_label, new_target, new_prev, new_next, new_sent, new_mwe])

    if label==1 and (mwe in prev):

        new_label = 1
        new_target = prev
        new_prev = ""
        new_next = target
        new_sent = new_prev + new_target + new_next
        new_mwe = mwe
        new_list.append([new_id, new_label, new_target, new_prev, new_next, new_sent, new_mwe])

    if label==1 and (mwe in next):

        new_label = 1
        new_target = next
        new_prev = target
        new_next = ""
        new_sent = new_prev + new_target + new_next
        new_mwe = mwe
        new_list.append([new_id, new_label, new_target, new_prev, new_next, new_sent, new_mwe])


for row in new_list:
    train = train.append(pd.Series(row, index=train.columns), ignore_index=True)

len(train)
train.isna().sum()
train.to_csv('/home/ojoo/SemEval-2022-Task2/MelBERT/My_Code_context copy/data/data_expand.csv', sep='\t', index=False) #tsv로 저장






########## CHECK ##########
n_1 = 0
n_2 = 0
n_3 = 0
n_4 = 0

for i in range(len(train)):

    ex = train.iloc[i,:]

    id = ex['ID'] # str
    label = ex['label'] # int
    target = ex['target'] # str
    prev = ex['previous'] # str
    next = ex['next'] # str
    sent = ex['sentence'] # str
    mwe = ex['mwe'] # str

    if label==0 and (mwe in prev):
        n_1 += 1
    if label==0 and (mwe in next):
        n_2 += 1
    if label==1 and (mwe in prev):
        n_3 += 1
    if label==1 and (mwe in next):
        n_4 += 1
    
print('idiomatic한 mwe가 prev에 포함되어있는 sample 수: ', n_1)
print('idiomatic한 mwe가 next에 포함되어있는 sample 수: ', n_2)
print('literal한 mwe가 prev에 포함되어있는 sample 수: ', n_3)
print('literal한 mwe가 next에 포함되어있는 sample 수: ', n_4)
print('총 개수: ', n_1 + n_2 + n_3 + n_4)

