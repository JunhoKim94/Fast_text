import numpy as np
from tqdm import tqdm
'''
word ==> subwords ==> summation(hidden) 이 추가된 word2vec = Fasttext
         BFS 이용subword 추출
'''
def make_corpus(data):
    '''
    Basic word2idx : ignore frequency, only word --> id
    data = number_of_data x 2 (title, description)
    '''

    words = []
    for lines in data:
        for line in lines:
            words += line.split()
        
    word2idx = {"UNK" : 0}
    for word in words:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    return word2idx

def word_to_id(data, word2idx):

    stack = []
    for lines in data:
        words = []
        for line in lines:
            temp = line.split()
            for word in temp:
                if word not in word2idx:
                    words += [word2idx["UNK"]]
                else:
                    words += [word2idx[word]]

        stack.append(words)

    length = [len(s) for s in stack]
    max_length = max(length)

    train_data = np.zeros((len(data), max_length + 1), dtype = np.int32)

    for i in tqdm(range(len(data))):
        train_data[i, :length[i]] = stack[i]
        train_data[i, -1] = length[i]

    return train_data


#def get_train_batch(data):


def make_ngram_corpus(n_grams):

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

if __name__ == "__main__":
    sub = make_corpus(3)
    corpus = corpus_to_hash(sub)
    x, sub_id = word2subwords("concatenate", 3, corpus)
    #sub_id = subwords2idx(x, corpus)
    print(x, sub_id)