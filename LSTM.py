import itertools
import getEmbeddings
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from collections import Counter
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from sklearn import metrics

# turning on the gpu use
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(allow_soft_placement=False)
config.gpu_options.allow_growth = True
# config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)


def plot_confusion_matrix(y_test, y_pred):
    normalize = False
    cmap = plt.cm.Blues
    classes = [1, 0]
    cm = metrics.confusion_matrix(y_test, y_pred, labels=classes)
    score = metrics.accuracy_score(y_test, y_pred)
    title = 'Confusion matrix Accuracy : ' + "{0:.2f}".format(score * 100)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def process_inp_news():
    fptr = open('y.txt','r')
    news1 = fptr.read()
    fptr.close()
    news1 = getEmbeddings.cleanup(news1)
    news1 = news1.split()
    if len(news1) < 10:
        return [" News cant be soo small !!",0]
    i = 0
    while i < len(news1):
        if news1[i] in word_bank:
            news1[i] = word_bank[news1[i]]
            i += 1
        else:
            del news1[i]
    news1 = sequence.pad_sequences([news1], maxlen=500)
    final_res = model.predict_classes(news1)[0][0]
    if final_res == 1:
        return ["Its a unreliable news !!",0]
    else:
        return ["Seems reliable...",1]


# initialize the program
top_words = 5000
epoch_num = 5
batch_size = 64
to_train = False
to_print_accuracy = False

# Read the text data
# 1 unreliable 0 reliable
if not os.path.isfile('./xtr_shuffled.npy') or \
        not os.path.isfile('./xte_shuffled.npy') or \
        not os.path.isfile('./ytr_shuffled.npy') or \
        not os.path.isfile('./yte_shuffled.npy'):
    getEmbeddings.clean_data()

xtr = np.load('./xtr_shuffled.npy')
xte = np.load('./xte_shuffled.npy')
y_train = np.load('./ytr_shuffled.npy')
y_test = np.load('./yte_shuffled.npy')

cnt = Counter()
x_train = []
for x in xtr:
    x_train.append(x.split())
    for word in x_train[-1]:
        cnt[word] += 1

    # Storing most common words
most_common = cnt.most_common(top_words + 1)
word_bank = {}
id_num = 1
for word, freq in most_common:
    word_bank[word] = id_num
    id_num += 1

# Encode the sentences
for news in x_train:
    i = 0
    while i < len(news):
        if news[i] in word_bank:
            news[i] = word_bank[news[i]]
            i += 1
        else:
            del news[i]

y_train = list(y_train)
y_test = list(y_test)

# Delete the short news
i = 0
while i < len(x_train):
    if len(x_train[i]) > 10:
        i += 1
    else:
        del x_train[i]
        del y_train[i]

# Generating test data
x_test = []
for x in xte:
    x_test.append(x.split())

# Encode the sentences
for news in x_test:
    i = 0
    while i < len(news):
        if news[i] in word_bank:
            news[i] = word_bank[news[i]]
            i += 1
        else:
            del news[i]

# Truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

# Convert to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Create the model
embedding_vecor_length = 32

if to_train:
    model = Sequential()
    model.add(Embedding(top_words + 2, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_num, batch_size=batch_size)
else:
    model = load_model('lstm.h5')

# Final evaluation of the model

process_inp_news()

if to_print_accuracy:
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy= %.2f%%" % (scores[1] * 100))

    # Draw the confusion matrix
    y_pred = model.predict_classes(X_test)
    plot_confusion_matrix(y_test, y_pred)

