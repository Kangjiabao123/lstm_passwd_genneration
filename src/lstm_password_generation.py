import sys
import random
import numpy as np
import keras.backend as K
from matplotlib import pyplot as plt
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from keras.models import Sequential


def sample(preds, temperature=1.0):
    """
    辅助采样函数，用于在predict函数返回的概率数组中进行采样。
    有关该函数作用请查看博客：https://www.jianshu.com/p/e054cd99089e
    :param preds: model.predict函数返回的概率数组
    :param temperature: 精度，建议尝试[0.2，0.5, 1.0, 1.2]查看效果，该参数越大，生成的文本将越open，反之越保守
    :return: 概率数组中采样索引
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)     # 多项式分布采样
    return np.argmax(probas)                        # 返回最大概率的值


"""
载入训练数据 -> 提取出涉及到的所有字符 -> 构成字典 """
raw_text = open("d:/passwd.txt", "r", encoding="utf-8").read().lower()
chars = sorted(list(set(raw_text)))
char2int = dict((c, i) for i, c in enumerate(chars))
int2char = dict((i, c) for i, c in enumerate(chars))

epoch = 10
batch_size = 256
maxlen = 50             # 每条句子的最大长度
sentences = []          # 所有句子
next_chars = []         # 预测后的下个字符

"""
对原始文本切片，构成神经网络的输入和输出"""
for i in range(0, len(raw_text) - maxlen, 3):
    sentences.append(raw_text[i: i + maxlen])
    next_chars.append(raw_text[i + maxlen])


"""
向量化数据，对输入输出进行one-hot编码"""
x_train = np.zeros((len(sentences), maxlen, len(chars)))
y_train = np.zeros((len(sentences), len(chars)))
for i, sentence in enumerate(sentences):    # 对于所有句子中的每一句
    y_train[i, char2int[next_chars[i]]] = 1
    for t, char in enumerate(sentence):     # 对于每一句中的每个字符
        x_train[i, t, char2int[char]] = 1


def on_epoch_end(epoch, _):
    """
    模型每次迭代完毕的回调函数
    :param epoch: 迭代次数
    :return: 无
    """
    print("------ Generating text after Epoch: ", epoch)
    model.save("G:/ABC/passwd_" + str(epoch) + ".model")

    """
    随机选取一段文本作文预测输入"""
    start_index = random.randint(0, len(raw_text) - maxlen)

    """
    分布式采样选取四个精度， 采样精度越大， 模型的输出越open"""
    temperature = [0.2, 0.5, 1.0, 1.2, 1.5]
    for temp in temperature:
        print()
        print("------ Temperature: ", temp)
        generrated = ""
        sentence = raw_text[start_index: start_index + maxlen]
        generrated += sentence

        print("------ Input: \n", sentence)
        print("------ Output:")

        """
        预测输出400个字符"""
        for i in range(400):

            """
            将输入转换为one-hot编码，用于输入到模型中"""
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char2int[char]] = 1

            """
            预测下一个字符的输出概率"""
            predicts = model.predict(x_pred, verbose=0)[0]

            """
            分布式采样"""
            next_index = sample(predicts, temp)

            """
            将预测到的字符索引转为其代表的字符，构成下次循环所用到的起始文本，
            并将当前预测到的字符输出到控制台"""
            next_char = int2char[next_index]
            generrated += next_char
            sentence = sentence[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
    return


print("Build model & train...")
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation="softmax"))
model.compile(loss="categorical_crossentropy",  optimizer=RMSprop(lr=0.01),  metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train,
                    epochs=epoch,
                    batch_size=batch_size,
                    callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])


def plot(savepath, title, xlabel, ylabel, type):
    """
    绘制训练精度或损失走势图
    :param savepath: 保存路径文件名
    :param title: 图片标题
    :param xlabel: 图片x轴标签
    :param ylabel: 图片y轴标签
    :param type: 类型：acc 或 loss
    :return: 无
    """
    fig = plt.figure()
    plt.plot(history.history[type])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([type], loc="best")
    fig.savefig(savepath)
    fig.clear()
    return


plot("../pic/model_accuracy.png", "model acc", "epoch", "accuracy", "acc")
plot("../pic/model_loss.png", "model loss", "epoch", "loss", "loss")
K.clear_session()
