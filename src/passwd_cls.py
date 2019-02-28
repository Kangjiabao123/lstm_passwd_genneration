from __future__ import print_function
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
import keras.backend as K
from keras.preprocessing.text import Tokenizer

# set parameters:
max_features = 5000
maxlen = 16
batch_size = 128
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10

print('Loading data...')
x_train = []
y_train = []
x_test = []
y_test = []

with open("d:/cls_charspasswd.txt", "r", encoding="utf-8") as f:
    while True:
        line = f.readline()
        if not line:
            break
        x_train.append(line.replace("\n", ""))
        y_train.append(0)
    f.close()

with open("d:/cls_numpasswd.txt", "r", encoding="utf-8") as f:
    while True:
        line = f.readline()
        if not line:
            break
        x_train.append(line.replace("\n", ""))
        y_train.append(1)
    f.close()

item_num = 10000
x_test = x_train[0:item_num] + x_train[len(x_train)-item_num:]
y_test = y_train[0:item_num] + y_train[len(y_test)-item_num:]

print('Pad sequences (samples x time)')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, maxlen=maxlen)
tokenizer.fit_on_texts(x_test)
x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, maxlen=maxlen)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

res = model.evaluate(x_test, y_test, batch_size)
print(res)
K.clear_session()
