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

number = 5

#path = "./data/ag_news/train.csv"
path = "C:/Users/dilab/Desktop/A. Multi-Class/yahoo_answers_csv/train.csv"
#path2 = "./data/ag_news/test.csv"
path2 = "C:/Users/dilab/Desktop/A. Multi-Class/yahoo_answers_csv/test.csv"
#split_file(path, number)

with open("./corpus.pickle", 'rb') as f:
    word2idx = pickle.load(f)

#word2idx  =  make_corpus(path)
test_data, test_label = get_words(path2)
test_data = word_to_id(test_data, word2idx, test_label)

vocab_size = len(word2idx)
class_num = 10
epochs = 10
learning_rate = 0.001
batch_size = 300
file_path = [path[:-4] + "%d.csv"%i for i in range(number)]


def train_np(vocab_size, class_num, epochs, learning_rate, file_path, test_data, word2idx):
    learning_rate = 0.001
    model = Fasttext(input_size = vocab_size, embed_size = 300, hidden = 10, output = class_num, padding_idx = None)
    #optimizer = SGD(lr = 0.001)
    criterion = Ce_losswithsoftmax()

    st = time.time()

    epoch_loss = 0
    loss_stack = []
    acc_stack = []

    for epoch in range(epochs + 1):
        learning_rate *= (0.95)**(epoch)
        for path in file_path:
            train_data,  label = get_words(path)
            train_data = word_to_id(train_data, word2idx, label)
            total_word = len(train_data)
            
            for iteration in range(len(train_data)):
                length = train_data[iteration,-2]
                x_train = train_data[iteration,:length]
                y_train = train_data[iteration,-1] - 1

                    
                y_pred = model.forward(x_train)
                loss = criterion.forward(y_pred, y_train)
                epoch_loss += loss

                d_out = criterion.backward()
                model.backward(d_out, learning_rate)

                #optimizer.update(model.params, model.grads)
                #optimizer._zero_grad(model.grads)
            
            epoch_loss /= len(train_data)
            loss_stack.append(epoch_loss)
            score = evaluate(test_data, model)
            acc_stack.append(score)

            if (epoch % 1 == 0):
                test_score = evaluate(test_data, model)
                curr_time = time.time()
                print(f"loss = {epoch_loss}  |  epoch  = {epoch}  | total_word = {total_word}  | time_spend = {curr_time - st} | val_score = {score}  | lr = {learning_rate}")

    return acc_stack, loss_stack

def train_torch(vocab_size, class_num, epochs, batch_size, learning_rate, file_path, test_data, word2idx):
    learning_rate = 0.001
    model = Fasttext_torch(vocab_size, embed_size= 300, hidden = 10, output= class_num, padding_idx= None)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    model.to(device)
    model.train()

    st = time.time()
    epoch_loss = 0
    loss_stack = []
    acc_stack = []
    tot = 0
    for epoch in range(epochs + 1):
        for path in file_path:
            train_data,  label = get_words(path)
            train_data = word_to_id(train_data, word2idx, label)
            total_word = len(train_data)
            tot += total_word
            for iteration in range(len(train_data)// batch_size):
                x_train, y_train = get_mini_pad(train_data, batch_size)

                x_train = torch.Tensor(x_train).to(torch.long).to(device)
                y_train = torch.Tensor(y_train).to(torch.long).to(device)

                y_pred = model(x_train)
                optimizer.zero_grad()
                loss = criterion(y_pred, y_train)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                if iteration % 100 == 0:
                    print(f"loss = {loss.item()}  |  iteration : {iteration}  | total iteration : {total_word // batch_size}")
        epoch_loss /= tot+1
        loss_stack.append(epoch_loss)
        
        x_test, y_test = get_mini_pad(test_data, batch_size)
        x_test, y_test = torch.Tensor(x_test).to(torch.long).to(device), torch.Tensor(y_test).to(torch.long).to(device)
        y_val  = torch.argmax(torch.nn.functional.softmax(model(x_test),dim = 1),dim = 1)
        score = len(y_test[y_test == y_val]) / len(y_test)
        #score = evaluate(test_data, model)
        #acc_stack.append(score)

        if (epoch % 1 == 0):
            #test_score = evaluate(test_data, model)
            curr_time = time.time()
            print(f"loss = {epoch_loss}  |  epoch  = {epoch}  | total_word = {total_word}  | time_spend = {curr_time - st} | val_score = {score}  | lr = {learning_rate}")
    
    return acc_stack, loss_stack

if __name__ == "__main__":
    #acc_stack, loss_stack = train_torch(vocab_size, class_num, epochs, batch_size, learning_rate, file_path, test_data, word2idx)
    acc_stack, loss_stack = train_np(vocab_size, class_num, epochs, learning_rate, file_path, test_data, word2idx)
    plot(acc_stack, loss_stack, epochs)