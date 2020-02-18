import numpy as np
import pandas as pd
from preprocess import *
from optim.optimizer import SGD
from model.model import Fasttext
from model.layers import Ce_losswithsoftmax
import time
import matplotlib.pyplot as plt
from utils import plot, evaluate

number = 5

#path = "./data/ag_news/train.csv"
path = "C:/Users/dilab/Desktop/A. Multi-Class/yahoo_answers_csv/train.csv"
#path2 = "./data/ag_news/test.csv"
path2 = "C:/Users/dilab/Desktop/A. Multi-Class/yahoo_answers_csv/test.csv"
#split_file(path, number)

data = pd.read_csv(path2, header = None)
data = data.fillna(" ")
test_label = np.array(data.iloc[:,0])
test_data = np.array(data.iloc[:,1:])
word2idx  =  make_corpus(path)
test_data = word_to_id(test_data, word2idx, test_label)

vocab_size = len(word2idx)
class_num = 4
epochs = 10
lr = 0.001
batch_size = 100
file_path = [path[:-4] + "%d.csv"%i for i in range(number)]


model = Fasttext(input_size = vocab_size, embed_size = 300, hidden = 10, output = class_num, padding_idx = None)
#optimizer = SGD(lr = 0.001)
criterion = Ce_losswithsoftmax()

st = time.time()
epoch_loss = 0
loss_stack = []
acc_stack = []
for epoch in range(epochs + 1):
    lr *= (0.95)**(epoch)
    for path in file_path:
        train_data,  label = get_train_words(path)
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
            model.backward(d_out, lr)

            #optimizer.update(model.params, model.grads)
            #optimizer._zero_grad(model.grads)
        
        epoch_loss /= len(train_data)
        loss_stack.append(epoch_loss)
        score = evaluate(test_data, model)
        acc_stack.append(score)

        if (epoch % 1 == 0):
            test_score = evaluate(test_data, model)
            curr_time = time.time()
            print(f"loss = {epoch_loss}  |  epoch  = {epoch}  | total_word = {total_word}  | time_spend = {curr_time - st} | val_score = {score}  | lr = {lr}")


plot(acc_stack, loss_stack, epochs)
