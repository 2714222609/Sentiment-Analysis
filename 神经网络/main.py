# -*- coding: utf-8 -*-

import codecs
import csv
from itertools import dropwhile

import jieba
import numpy as np
import pandas as pd

import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Bidirectional, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout

cpu_count = multiprocessing.cpu_count()  # 4
vocab_dim = 100
n_iterations = 1
n_exposures = 10  # 所有频数超过10的词语
window_size = 7
n_epoch = 10
maxlen = 100
batch_size = 32


def loadfile():
    neg = pd.read_csv('data/neg_train.csv', header=None, index_col=None, sep='\t')
    pos = pd.read_csv('data/pos_train.csv', header=None, index_col=None, sep='\t')

    combined = np.concatenate((pos[0],neg[0]))
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))

    return combined, y

def tokenizer(data):
    text = [jieba.lcut(document.replace('\n', '')) for document in data]
    return text


def create_dictionaries(model=None, combined=None):

    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),#将词库里的词打上索引，构建一个索引词库
                            allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 将索引词库转换为字典形式
        f = open("word2index.txt", 'w', encoding='utf8')
        for key in w2indx:
            f.write(str(key))
            f.write(' ')
            f.write(str(w2indx[key]))
            f.write('\n')
        f.close()
        w2vec = {word: model[word] for word in w2indx.keys()}  # 构建词和词向量的映射，存在词典中


        def parse_dataset(combined):  # 用索引代替词表示
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except Exception:
                        new_txt.append(0)  # 没有的就将索引设置为0
                data.append(new_txt)
            return data  # word=>index

        combined = parse_dataset(combined)  # [[1,2,3...],[]]
        combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 规范长度，所有程度与最大长度对齐
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


# 创建词语字典，并返回每个词语的索引，词与词向量，以及所有句子所对应的词语索引
def word2vec_train(combined):
    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)  # input: list
    model.train(combined, total_examples=model.corpus_count, epochs=model.iter)
    model.save('./model/Word2vec_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined #带有索引的词库，所有词的词向量库，将所有句子分词用索引代替


def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2,random_state=5)
    y_train = keras.utils.to_categorical(y_train, num_classes=2)  # 转换为对应one-hot 表示  [len(y),2]
    y_test = keras.utils.to_categorical(y_test, num_classes=2)
    # print (x_train.shape,y_train.shape)
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test
    #n_symbols：词库的长度
    #embedding_weights：索引和词向量的映射词典


##定义网络结构
def bilstm(n_symbols, embedding_weights, x_train, y_train,x_test,y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,#embedding_size
                        input_dim=n_symbols,#vocab_dim
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=maxlen))  # Adding Input Length
    # model.add(LSTM(output_dim=50, activation='tanh'))
    model.add(Bidirectional(LSTM(units=50, activation='tanh')))
    model.add(Dropout(0.5))#避免过拟合，按比例丢掉部分单元
    model.add(Dense(2, activation='softmax'))  # Dense=>全连接层,输出维度=2

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=2)
    y_pred = model.predict(x_test)

    for i in range(len(y_pred)):
        max_value = max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value == y_pred[i][j]:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0
    #target_names = (['负面', '正面'])
    print(classification_report(y_test, y_pred))
    # model.save('./model/bilstm.h5')

def lstm(x_train, y_train, x_test, y_test):
    model = Sequential()  # Sequential 序贯模型，net通过ADD加入
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=maxlen))

    lstmCell=LSTM(units=50, dropout=0.25, recurrent_dropout=0.25)
    model.add(lstmCell)
    model.add(Dense(2))  #全连接层
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    y_pred = model.predict(x_test)

    for i in range(len(y_pred)):
        max_value = max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value == y_pred[i][j]:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0
    # target_names = (['负面', '正面']
    print(classification_report(y_test, y_pred))


def dnn(x_train, y_train, x_test, y_test):
 # 定义模型
    init = keras.initializers.glorot_uniform(seed=1)
    simple_adam = keras.optimizers.Adam()
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=50, input_dim=100, kernel_initializer=init, activation='relu'))
    model.add(keras.layers.Dense(units=50, kernel_initializer=init, activation='relu'))
    model.add(keras.layers.Dense(units=2, kernel_initializer=init, activation='softmax'))
    # model.add(Dropout(0.25))
    model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

 # 训练模型
    b_size = 32
    max_epochs = 30
    print("Starting training ")
    h = model.fit(x_train, y_train, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=2)
    print("Training finished \n")

 # 评估模型
    eval = model.evaluate(x_test, y_test, verbose=2)
    # print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
    #       % (eval[0], eval[1] * 100) )
    y_pred = model.predict(x_test)

    for i in range(len(y_pred)):
        max_value = max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value == y_pred[i][j]:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0
    # target_names = (['负面', '正面']
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    # 训练模型，并保存
    print('加载数据集...')
    combined, y = loadfile()
    print(len(combined), len(y))
    print('数据预处理...')
    combined = tokenizer(combined)
    print('训练word2vec模型...')
    index_dict, word_vectors, combined = word2vec_train(combined)
    # print("index_dic:",index_dict,'/n')
    # print("word_vectors:",word_vectors,'/n')
    # print("combined:",combined,'/n')

    print('将数据转换为模型输入所需格式...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined,
                                                                              y)
    print("x:",x_train)
    print("________________________")
    print("y:",y_train)


    print("特征与标签大小:")
    print("x:",x_train.shape)
    print("________________________")
    print("\ny:",y_train.shape)

    # dnn(x_train, y_train, x_test, y_test)

    # print('训练bilstm模型...')
    # bilstm(n_symbols, embedding_weights, x_train, y_train,x_test,y_test)

    lstm(x_train, y_train, x_test, y_test)

    # print('加载bilstm模型...')
    # model = load_model('./model/bilstm.h5')
    # y_pred = model.predict(x_test)

    # for i in range(len(y_pred)):
    #     max_value = max(y_pred[i])
    #     for j in range(len(y_pred[i])):
    #         if max_value == y_pred[i][j]:
    #             y_pred[i][j] = 1
    #         else:
    #             y_pred[i][j] = 0
    # # target_names = (['负面', '正面']
    # print(classification_report(y_test, y_pred))