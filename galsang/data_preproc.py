import pandas as pd
import numpy as np
import os
import csv
import re
import string
from pathlib import Path


def load_csv(path):
    header = None
    data = list()
    with open(path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if header is None:
                header = row
                continue
            data.append(row)
    return header, data


def write_csv(data, location):
    with open(location, 'w', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    print("Wrote {}".format(location))


def _get_train_data(data_location, file_name, include_context, include_idiom):
    file_name = os.path.join(data_location, file_name)

    header, data = load_csv(file_name)

    out_header = ['ID', 'label', 'target', 'previous', 'next', 'sentence']
    if include_idiom:
        out_header = ['ID', 'label', 'target', 'previous', 'next', 'sentence', 'mwe']

    # train: ['DataID', 'Language', 'MWE', 'Setting', 'Previous', 'Target', 'Next', 'Label']
    out_data = list()
    for elem in data:
        label = elem[header.index('Label')]
        sentence = elem[header.index('Target')]
        target = elem[header.index('Target')]
        id = elem[header.index('DataID')]
        prev = elem[header.index('Previous')]
        next = elem[header.index('Next')]
        if include_context:
            sentence = ' '.join(
                [elem[header.index('Previous')], elem[header.index('Target')], elem[header.index('Next')]])
        this_row = None
        if not include_idiom:
            this_row = [id, label, target, prev, next, sentence]
        else:
            mwe = elem[header.index('MWE')]
            this_row = [id, label, target, prev, next, sentence, mwe]
        out_data.append(this_row)
        assert len(out_header) == len(this_row)
    return [out_header] + out_data


def _get_dev_eval_data(data_location, input_file_name, gold_file_name, include_context, include_idiom):
    input_headers, input_data = load_csv(os.path.join(data_location, input_file_name))
    gold_header = gold_data = None
    if not gold_file_name is None:
        gold_header, gold_data = load_csv(os.path.join(data_location, gold_file_name))
        assert len(input_data) == len(gold_data)

    # dev, eval: ['ID', 'Language', 'MWE', 'Previous', 'Target', 'Next']
    # gold: ['ID', 'DataID', 'Language', 'Label']

    out_header = ['ID', 'label', 'target', 'previous', 'next', 'sentence']
    if include_idiom:
        out_header = ['ID', 'label', 'target', 'previous', 'next', 'sentence', 'mwe']

    out_data = list()
    for index in range(len(input_data)):
        label = 1  # gold 값이 없는 경우 모두 1
        if not gold_file_name is None:
            this_input_id = input_data[index][input_headers.index('ID')]
            this_gold_id = gold_data[index][gold_header.index('ID')]
            assert this_input_id == this_gold_id

            label = gold_data[index][gold_header.index('Label')]

        elem = input_data[index]
        sentence = elem[input_headers.index('Target')]
        id = elem[input_headers.index('ID')]
        target = elem[input_headers.index('Target')]
        prev = elem[input_headers.index('Previous')]
        next = elem[input_headers.index('Next')]

        if include_context:
            sentence = ' '.join([elem[input_headers.index('Previous')], elem[input_headers.index('Target')],
                                 elem[input_headers.index('Next')]])
        this_row = None
        if not include_idiom:
            this_row = [id, label, target, prev, next, sentence]
        else:
            mwe = elem[input_headers.index('MWE')]
            this_row = [id, label, target, prev, next, sentence, mwe]
        assert len(out_header) == len(this_row)
        out_data.append(this_row)

    return [out_header] + out_data


def create_data(input_location, output_location):
    ## Zero shot data
    train_data = _get_train_data(
        data_location=input_location,
        file_name='train_zero_shot.csv',
        include_context=True,
        include_idiom=True
    )
    write_csv(train_data, os.path.join(output_location, 'train.csv'))

    dev_data = _get_dev_eval_data(
        data_location=input_location,
        input_file_name='dev.csv',
        gold_file_name='dev_gold.csv',
        include_context=True,
        include_idiom=True
    )
    write_csv(dev_data, os.path.join(output_location, 'dev.csv'))

    eval_data = _get_dev_eval_data(
        data_location=input_location,
        input_file_name='eval.csv',
        gold_file_name=None,  ## Don't have gold evaluation file -- submit to CodaLab
        include_context=True,
        include_idiom=True
    )
    write_csv(eval_data, os.path.join(output_location, 'eval.csv'))

    '''
    ## OneShot Data (combine both for training)
    train_zero_data = _get_train_data(
        data_location   = input_location,
        file_name       = 'train_zero_shot.csv',
        include_context = True,
        include_idiom   = True
    )
    
    train_one_data = _get_train_data(
        data_location   = input_location,
        file_name       = 'train_one_shot.csv',
        include_context = True,
        include_idiom   = True
    )

    assert train_zero_data[0] == train_one_data[0] ## Headers
    train_data = train_one_data + train_zero_data[1:]
    write_csv( train_data, os.path.join( output_location, 'OneShot', 'train.csv' ) )
    
    dev_data = _get_dev_eval_data(
        data_location    = input_location,
        input_file_name  = 'dev.csv',
        gold_file_name   = 'dev_gold.csv', 
        include_context  = True,
        include_idiom    = True
    )        
    write_csv( dev_data, os.path.join( output_location, 'OneShot', 'dev.csv' ) )
    
    eval_data = _get_dev_eval_data(
        data_location    = input_location,
        input_file_name  = 'eval.csv',
        gold_file_name   = None,
        include_context  = True,
        include_idiom    = True
    )
    
    write_csv( eval_data, os.path.join( output_location, 'OneShot', 'eval.csv' ) )
    '''


def save_data(input_location, output_location):
    # input_location = '/home/ojoo/SemEval-2022-Task2/MelBERT/My_Code_context/preproc/orig_data/'
    # output_location = '/home/ojoo/SemEval-2022-Task2/MelBERT/My_Code_context/preproc/preprec_data/'

    # Path( os.path.join( output_location, 'ZeroShot' ) ).mkdir(parents=True, exist_ok=True)
    # Path( os.path.join( output_location, 'OneShot' ) ).mkdir(parents=True, exist_ok=True)

    create_data(input_location, output_location)


def extend_data(input_location, output_location):
    train = pd.read_csv(input_location + 'train.csv', encoding='utf-8', sep=',')
    train = train.fillna('')

    new_list = []

    for i in range(len(train)):

        ex = train.iloc[i, :]

        id = ex['ID']  # str
        label = ex['label']  # int
        target = ex['target']  # str
        prev = ex['previous']  # str
        next = ex['next']  # str
        sent = ex['sentence']  # str
        mwe = ex['mwe']  # str

        new_id = id + '-1'

        if label == 0 and (mwe in prev):
            new_label = 0
            new_target = prev
            new_prev = ""
            new_next = target
            new_sent = new_prev + new_target + new_next
            new_mwe = mwe
            new_list.append([new_id, new_label, new_target, new_prev, new_next, new_sent, new_mwe])

        if label == 0 and (mwe in next):
            new_label = 0
            new_target = next
            new_prev = target
            new_next = ""
            new_sent = new_prev + new_target + new_next
            new_mwe = mwe
            new_list.append([new_id, new_label, new_target, new_prev, new_next, new_sent, new_mwe])

        if label == 1 and (mwe in prev):
            new_label = 1
            new_target = prev
            new_prev = ""
            new_next = target
            new_sent = new_prev + new_target + new_next
            new_mwe = mwe
            new_list.append([new_id, new_label, new_target, new_prev, new_next, new_sent, new_mwe])

        if label == 1 and (mwe in next):
            new_label = 1
            new_target = next
            new_prev = target
            new_next = ""
            new_sent = new_prev + new_target + new_next
            new_mwe = mwe
            new_list.append([new_id, new_label, new_target, new_prev, new_next, new_sent, new_mwe])

    for row in new_list:
        train = train.append(pd.Series(row, index=train.columns), ignore_index=True)

    train.to_csv(output_location + 'train.csv', sep=',', index=False)  # tsv로 저장


def make_index_col(df, output_location, filename):
    index_col_list = []

    for row in range(len(df)):

        sent = df['target'][row]
        # new_sent = re.sub('[=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sent)
        new_sent = re.sub('-', ' ', sent)
        # new_sent = new_sent.replace('.','')
        df['target'][row] = new_sent

        # df['previous'][row] = re.sub("'", '\'', str(df['previous'][row]))
        # df['target'][row] = re.sub("'", '\'', str(df['target'][row]))
        # df['next'][row] = re.sub("'", '\'', str(df['next'][row]))

        mwe = df['mwe'][row]
        # new_mwe = re.sub('[=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', mwe)
        new_mwe = re.sub('-', ' ', mwe)
        df['mwe'][row] = new_mwe

        mwe_list_ = new_mwe.split(" ")
        mwe_list = new_mwe.split(" ")

        for mwe_comp in mwe_list_:  # 대문자 추가
            mwe_comp = string.capwords(mwe_comp)
            mwe_list.append(mwe_comp)

        i_list = []
        mwe_index = []

        for i, value in enumerate(new_sent.split(' ')):
            for mwe in mwe_list:
                if mwe in value:
                    i_list.append(i)
        # i_list = [i for i, value in enumerate(new_sent.split(' ')) if value in mwe_list] 

        for i in range(len(i_list) - 1):
            if i_list[i + 1] - i_list[i] == 1:
                mwe_index = str(i_list[i]) + ' ' + str(i_list[i + 1])

        index_col_list.append(mwe_index)

    df['index'] = index_col_list
    # output_location = '/home/ojoo/SemEval-2022-Task2/MelBERT/My_Code_context/preproc/preprec_data/'
    df.to_csv(output_location + '{}.csv'.format(filename), sep='\t', index=False)  # tsv로 저장


def create_final_data(location):
    # input_location = '/home/ojoo/SemEval-2022-Task2/MelBERT/My_Code_context/preproc/preprec_data/ZeroShot/'
    train = pd.read_csv(location + 'train.csv', encoding='utf-8')
    dev = pd.read_csv(location + 'dev.csv', encoding='utf-8')
    eval = pd.read_csv(location + 'eval.csv', encoding='utf-8')

    make_index_col(train, location, 'train')
    make_index_col(dev, location, 'dev')
    make_index_col(eval, location, 'eval')
