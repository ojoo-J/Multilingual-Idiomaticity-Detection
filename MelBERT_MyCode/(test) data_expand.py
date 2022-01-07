import pandas as pd
import numpy as np
import os
import csv
import re
import string
from pathlib import Path

train = pd.read_csv('/home/ojoo/SemEval-2022-Task2/MelBERT/My_Code_context copy/data/preproc/train.csv', encoding='utf-8', sep='\t')
train.head()

#for i in range(len(train)):
    
ex = train.iloc[0,:]
id = ex['ID']
label = ex['label']
target = ex['target']
prev = ex['previous']
next = ex['next']
sent = ex['sentence']
mwe = ex['mwe']
