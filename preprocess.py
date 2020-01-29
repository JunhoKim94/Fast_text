import numpy as np

'''
word ==> subwords ==> summation(hidden) 이 추가된 word2vec = Fasttext
         BFS 이용subword 추출
'''

def make_corpus(n_grams):

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