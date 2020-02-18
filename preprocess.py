import numpy as np
from tqdm import tqdm
import pandas as pd
import collections
import re

'''
word ==> subwords ==> summation(hidden) 이 추가된 word2vec = Fasttext
         BFS 이용subword 추출
'''
def split_file(path, number):
    data = pd.read_csv(path, header = None, encoding= 'utf-8')
    data = data.fillna(" ")

    total = len(data)
    batch = total // number

    for i in range(number):
        data.iloc[i *batch : (i+1) * batch + 1].to_csv(path[:-4] + "%d.csv"%i, header = False, index = False) 

def make_corpus(path):
    '''
    Basic word2idx : ignore frequency, only word --> id
    data = number_of_data x 2 (title, description)
    
    #data = pd.read_csv(path, header = None, encoding= 'utf-8')
    #data = data.fillna(" ")

    #label = np.array(data.iloc[:,0])
    #train_data = np.array(data.iloc[:,1:])
    
    words = []
    for lines in tqdm(train_data):
        for line in lines:
            words += line.split()
    '''
    words = []
    with open(path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            temp = line[1:].split("\",\"")
            for sen in temp[1:]:
                sen = sen.strip()
                sen = clean_str(sen, True)                
                words += sen.split()
        
    word2idx = {"UNK" : 0}
    for word in words:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
            
    return word2idx


def word_to_id(data, word2idx, label):
    '''
    data = ["title", "description"] => (total_size, 1, 1)
    train_data = [word_id + length + class] => (total_size, max_length + 2)
    '''
    stack = []
    for lines in data:
        words = []
        for line in lines:
            temp = line.split()
            for word in temp:
                if word not in word2idx:
                    #words += [word2idx["UNK"]]
                    continue
                else:
                    words += [word2idx[word]]

        stack.append(words)

    length = [len(s) for s in stack]
    max_length = max(length)

    train_data = np.zeros((len(data), max_length + 2), dtype = np.int32)

    for i in tqdm(range(len(data))):
        train_data[i, :length[i]] = stack[i]
        train_data[i, -2] = length[i]
        train_data[i, -1] = label[i]

    return train_data

def get_train_words(path):
    train_word = []
    label = []
    with open(path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            label += [int(line[0])]
            line = line[2:].strip()
            line = clean_str(line, True)
            train_word.append(line.split())

    return train_word, np.array(label)

def get_mini_pad(train_data, target, batch_size):

    seed = np.random.choice(len(train_data), batch_size)
    batch_data = train_data[seed, :]
    length = batch_data[seed,-1]

    max_length = max(length)

    batch_data = batch_data[:, :max_length]
    
    return batch_data, target[seed]


def make_subwords_corpus(n_grams):

    def make_subword(n_grams):
        alphabet = [chr(i) for i in range(ord('a'),ord('z')+1)]

        stack = [chr(i) for i in range(ord('a'),ord('z')+1)]
        while(1):
            temp = stack.pop(0)
            if len(temp) >= n_grams:
                break

            for alpha in alphabet:
                new_alpha = temp + alpha
                stack.append(new_alpha)

        return stack

    stack = make_subword(n_grams)

    if n_grams > 1:
        add_1 = make_subword(n_grams - 1)
        for al in add_1:
            stack.append("<" + al)
            stack.append(al + ">")

        if n_grams > 2:
            add_2 = make_subword(n_grams - 2)
            for al in add_2:
                stack.append("<" + al + ">")

    return stack

def corpus_to_hash(word):
    corpus = dict()
    for i in range(len(word)):
        corpus[word[i]] = i

    return corpus
        
def word2subwords(word, n_grams, corpus):

    temp = "<" + word + ">"
    sub_words = []
    ret = []
    for i in range(len(temp) - n_grams + 1):
        sub_words.append(temp[i:i+n_grams])
        ret.append(corpus[temp[i:i+n_grams]])

    return sub_words, ret

def subwords2idx(sub_words, corpus):
    
    ret = []
    for sub in sub_words:
        ret.append(corpus[sub])

    return ret

def gen_train(data, val_ratio = 0.1):
    np.random.shuffle(data)
    num = int(val_ratio * len(data))

    return data[num:], data[:num]

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()