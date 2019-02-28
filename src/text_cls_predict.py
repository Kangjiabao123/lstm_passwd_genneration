import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


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


# 所有分类
cls_dict = {0: "汽车", 1: "文化", 2: "经济", 3: "医学", 4: "军事", 5: "运动"}

maxlen = 100
sentences_train, y_train = load_dataset()
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
x_train = tokenizer.texts_to_sequences(sentences_train)
vocab_size = len(tokenizer.word_index) + 1
print("vocab_size: ", vocab_size)
x_train = pad_sequences(x_train, padding="post", maxlen=maxlen)
print(x_train.shape)


model = load_model("../model/text_cls.mod")
predict_result = model.predict_classes(x_train[:100], 1)
print(predict_result)
keras.backend.clear_session()

