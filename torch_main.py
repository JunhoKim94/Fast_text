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

print("\n ==============================> Training Start <=============================")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(torch.cuda.is_available())

n_grams = False

path = "./data/sogou_news_csv/train.csv"
#path = "./data/ag_news_csv/train.csv"
path2 = "./data/sogou_news_csv/test.csv"
#path2 = "./data/ag_news_csv/test.csv"
#split_file(path, number)

word2idx, data, label = make_corpus(path, n_grams)

test_data, test_label = get_sentence(path2)
test_data = word_to_id(test_data, test_label , word2idx, n_grams)

vocab_size = len(word2idx)
class_num = 5
epochs = 10
learning_rate = 0.1
batch_size = 512
max_len = 1500

model = Fasttext_torch(vocab_size, embed_size = 10, hidden = 10, output= class_num, padding_idx= None)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = torch.nn.CrossEntropyLoss()
#torch.nn.utils.clip_grad_norm_(model.parameters(), 9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

model.to(device)
model.train()

st = time.time()
epoch_loss = 0
loss_stack = []
acc_stack = []

#data, label = get_sentence(path)
total_word = len(data)

def scor(test_data):
    score = 0
    model.eval()
    for i in range(len(test_data) // batch_size):
        x_test, y_test = get_mini_pad(test_data, batch_size, max_len)
        x_test, y_test = torch.LongTensor(x_test).to(device), torch.LongTensor(y_test).to(device)
        y_val = torch.argmax(torch.nn.functional.softmax(model(x_test), dim = 1), dim = 1)
        score += len(y_test[y_test == y_val])

    score /= len(test_data)
    return score

for epoch in range(epochs + 1):
    learning_rate *= (0.95)**(epoch)
    
    #메모리 할당을 위한 batch
    
    epoch_loss = 0
    model.train()
    for iteration in range(total_word // batch_size):
        
        b_data, b_label = rand_sample(data, label, batch_size)
        #train_data = word_to_id(data[iteration * batch_size : (iteration + 1)*batch_size], label[iteration * batch_size : (iteration + 1)*batch_size], word2idx, n_grams)
        train_data = word_to_id(b_data, b_label, word2idx, n_grams)
        x_train, y_train = get_mini_pad(train_data, batch_size, max_len)

        x_train = torch.Tensor(x_train).to(torch.long).to(device)
        y_train = torch.Tensor(y_train).to(torch.long).to(device)

        y_pred = model(x_train)
        optimizer.zero_grad()
        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        y = torch.argmax(torch.nn.functional.softmax(y_pred, dim = 1), dim = 1)
        t_score = len(y_train[y_train == y]) / len(y_train)


    epoch_loss /= total_word+1
    loss_stack.append(epoch_loss)
    '''
    x_test, y_test = get_mini_pad(test_data, batch_size, 300)
    x_test, y_test = torch.Tensor(x_test).to(torch.long).to(device), torch.Tensor(y_test).to(torch.long).to(device)
    y_val  = torch.argmax(torch.nn.functional.softmax(model(x_test),dim = 1),dim = 1)
    score = len(y_test[y_test == y_val]) / len(y_test)
    #score = evaluate(test_data, model)
    #acc_stack.append(score)
    '''
    score = scor(test_data)


    if (epoch % 1 == 0):
        #test_score = evaluate(test_data, model)
        curr_time = time.time()
        print(f"loss = {epoch_loss}  |  epoch  = {epoch}  | total_word = {total_word}  | time_spend = {curr_time - st} | val_score = {score}  | lr = {learning_rate} | t_score : {t_score}")
        del epoch_loss, epoch, x_train, y_train, y_pred, loss
