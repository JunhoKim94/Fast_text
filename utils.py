import numpy as np
import matplotlib.pyplot as plt


def plot(acc_stack, loss_stack, epochs):
    a = [i for i in range(epochs + 1)]
    
    plt.figure(figsize = (10,8))
    fig , ax1 = plt.subplots()
    ax2 = ax1.twinx()
    acc = ax1.plot(a, acc_stack, 'r', label = 'Accuracy')
    loss = ax2.plot(a, loss_stack, 'b', label = 'loss')
    plt.legend()
    ax1.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax1.set_ylabel("accuracy")

    ax = acc + loss
    labels = [l.get_label() for l in ax]
    plt.legend(ax, labels, loc =2)

    plt.show()
    
def evaluate(dev, model):
    total = len(dev)
    score = 0

    for i in range(total):
        length = dev[i, -2]
        x = dev[i, :length]
        y = dev[i, -1] - 1

        y_val = np.argmax(model.forward(x), axis = 1) 
        if y == y_val[0]:
            score += 1

    return score / total