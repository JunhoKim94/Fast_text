import numpy as np
import pandas as pd
from preprocess import *
from optim.optimizer import SGD
from model.model import Fasttext, Fasttext_torch
from model.layers import Ce_losswithsoftmax
import time
import matplotlib.pyplot as plt
from utils import plot, evaluate
import torch
import pickle
import random

n_grams = False

#path = "./data/yahoo_answers_csv/train.csv"
path = "./data/ag_news_csv/train.csv"
#path2 = "./data/yahoo_answers_csv/test.csv"
path2 = "./data/ag_news_csv/test.csv"

word2idx, data, label = make_corpus(path, n_grams)

test_data, test_label = get_sentence(path2)
#test_data = word_to_id(test_data, test_label , word2idx, n_grams)

vocab_size = len(word2idx)
class_num = 4
epochs = 10
learning_rate = 0.001
batch_size = 64

model = Fasttext(input_size = vocab_size, embed_size = 10, hidden = 10, output = class_num, padding_idx = None)
#optimizer = SGD(lr = 0.001)
criterion = Ce_losswithsoftmax()

st = time.time()

loss_stack = []
acc_stack = []
#data, label = get_sentence(path)
total_word = len(data)

for epoch in range(epochs + 1):
    learning_rate *= (0.95)**(epoch)
    
    #메모리 할당을 위한 batch
    epoch_loss = 0
    for iteration in range(total_word):
        seed = random.randint(0, total_word)
        train_data = word_to_id([data[seed]], [label[seed]], word2idx, n_grams)
        length = train_data[0,-2]
        x_train = train_data[0,:length]
        y_train = train_data[0,-1] - 1

        y_pred = model.forward(x_train)
        loss = criterion.forward(y_pred, y_train)
        epoch_loss += loss

        d_out = criterion.backward()
        model.backward(d_out, learning_rate)

        #optimizer.update(model.params, model.grads)
        #optimizer._zero_grad(model.grads)

    epoch_loss /= len(data)
    loss_stack.append(epoch_loss)
    score = evaluate(test_data, test_label, model, word2idx, n_grams)
    acc_stack.append(score)

    if (epoch % 1 == 0):
        #test_score = evaluate(test_data, model)
        curr_time = time.time()
        print(f"loss = {epoch_loss}  |  epoch  = {epoch}  | total_word = {total_word}  | time_spend = {curr_time - st} | val_score = {score}  | lr = {learning_rate}")


plot(acc_stack, loss_stack, epochs)
