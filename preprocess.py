import numpy as np
from tqdm import tqdm
import pandas as pd
import collections
import re
import pickle

def make_corpus(path, n_grams = None):
    '''
    Basic word2idx : ignore frequency, only word --> id
    data = number_of_data x n (title, description, ...)
    '''
    words = []
    label = []
    with open(path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            sen = line[4:]
            #sen = sen.strip()
            sen = clean_str(sen, True)            
            words += [sen]
            label += [int(clean_str(line[:4]))]
    word2idx = {"UNK" : 0}
    for line in words:
        line = line.split()
        if n_grams:
            line += ['%s_%s' % (line[index],line[index+1]) for index in range(len(line)-1)]
        for word in line:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
        
    #with open("./corpus.pickle", "wb") as f:
    #    pickle.dump(word2idx,f,protocol=pickle.HIGHEST_PROTOCOL)
            
    return word2idx, words , label

def rand_sample(data, label, batch):
    seed = np.random.choice(len(data), batch)
    b_train = []
    b_label = []
    for s in seed:
        b_train.append(data[s])
        b_label.append(label[s])

    return b_train, b_label



def word_to_id(data, label, word2idx, n_grams = False):
    '''
    data = ["title", "description"] => (total_size, 1, 1)
    train_data = [word_id + length + class] => (total_size, max_length + 2)
    '''

    stack = []
    for lines in data:
        lines = lines.split()
        if len(lines) < 1:
            continue
        words = []
        if n_grams:
            lines += ['%s_%s' % (lines[index],lines[index+1]) for index in range(len(lines)-1)]
        for word in lines:
            if word in word2idx:
                words += [word2idx[word]]

        stack.append(words)

    length = [len(s) for s in stack]
    #print(length)
    max_length = max(length)

    train_data = np.zeros((len(stack), max_length + 2), dtype = np.int32)

    for i in range(len(stack)):
        train_data[i, :length[i]] = stack[i]
        train_data[i, -2] = length[i]
        train_data[i, -1] = label[i]
    return train_data

def get_sentence(path):
    train_word = []
    label = []
    with open(path, 'r', encoding = 'utf-8') as f:
        data = []
        label = []
        for line in tqdm(f.readlines()):
            label.append(int(clean_str(line[:4])))
            data.append(clean_str(line[4:]))

    return data, label

def get_mini_pad(train_data, batch_size, max_len):

    seed = np.random.choice(len(train_data), batch_size)
    batch_data = train_data[seed, :]
    length = batch_data[:,-2]
    target = batch_data[:,-1]
    
    max_length = max(length)
    if max_length > max_len:
        max_length = max_len
    #print(max_length)
    batch_data = batch_data[:, :max_length]
    
    return batch_data, target - 1


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

def clean_str(string, TREC=True):
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
    string = re.sub(r",", " ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\"", "", string)
    return string.strip() if TREC else string.strip().lower()