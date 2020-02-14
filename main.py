import numpy as np
import pandas as pd
from preprocess import *

path = "./data/ag_news/train.csv"
data = pd.read_csv(path, header = None)

label = np.array(data.iloc[:,0])
train_data = np.array(data.iloc[:,1:])

word2idx = make_corpus(train_data)
train_data = word_to_id(train_data, word2idx)
