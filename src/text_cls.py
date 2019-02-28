from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, Conv1D, Dense, Dropout, Activation, GlobalMaxPooling1D, MaxPool1D, Flatten, BatchNormalization
from matplotlib import pyplot as plt
import numpy as np
import keras.backend as K


def load_dataset():
    """
    加载6个分类的数据。
    每个文件是一个分类，总共六类。
    各文件中每行是一条训练数据，单文件内共4000条。
    :return: （数据， 标签）
    """

    datalines = []
    labels = []

    for i in range(6):
        fname = "D:/dataset/train/cls_" + str(i) + ".txt"
        with open(fname, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    f.close()
                    break
                datalines.append(line)
                labels.append(i)
    return datalines, labels


def load_data():
    x_train = np.load("../data/x_train.npy")
    y_train = np.load("../data/y_train.npy")
    return x_train, y_train


def Build_CNN():
    """
    效仿LeNet-5
    LeNet-5是卷积神经网络的作者Yann LeCun用于MNIST识别任务提出的模型。
    模型很简单，就是卷积池化层的堆叠，最后加上几层全连接层。我们依样画葫芦，
    将它运用在文本分类任务中，只是模型的输入不同。
    :return:
    """

    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length=maxlen))
    model.add(Conv1D(256, 3, padding="same"))
    model.add(MaxPool1D(3, 3, padding='same'))
    model.add(Conv1D(128, 3, padding='same'))
    model.add(MaxPool1D(3,3,padding='same'))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(BatchNormalization()) # (批)规范化层
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(class_num,activation='softmax'))
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    return model


def BuildCNN():
    embedding_dims = 50
    cnn_filters = 100
    cnn_kernel_size = 5

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dims, input_length=maxlen))
    model.add(Conv1D(cnn_filters, cnn_kernel_size, padding='same', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    return model


maxlen = 100
class_num = 6
embedding_dim = 100
batch_size = 32
epochs = 5

print("load data...")
# sentences_train, y_train = load_data()
sentences_train, y_train = load_dataset()

# 转one-hot编码
y_train = to_categorical(y_train, class_num)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
x_train = tokenizer.texts_to_sequences(sentences_train)
vocab_size = len(tokenizer.word_index) + 1
print("vocab_size: ", vocab_size)
x_train = pad_sequences(x_train, padding="post", maxlen=maxlen)
print(x_train.shape)


model = BuildCNN()     # Build_CNN()
history = model.fit(x_train, y_train, epochs=epochs, verbose=1, batch_size=batch_size, shuffle=True)
loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
print("\nTraining Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
# print("Testing Accuracy:  {:.4f}".format(accuracy))

model.save("../model/text_cls.mod")

# 绘制模型的精度和损失，可视化方便观察
fig = plt.figure()

plt.plot(history.history['loss'])
plt.xlabel("epoch")
plt.title('model loss')
plt.legend(["loss"], loc='best')
plt.show()
fig.savefig('../pic/cls_loss.png')
fig.clear()

plt.plot(history.history['acc'])
plt.xlabel('epoch')
plt.title('model accuracy')
plt.legend(['accuracy'], loc='best')
plt.show()
fig.savefig('../pic/cls_accuracy.png')
fig.clear()

# 清理
K.clear_session()
