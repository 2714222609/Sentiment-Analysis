from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pickle
import jieba
import os
import re
import string

# 20000条训练集
file_path1 = '../dataset/train/neg.txt'
file_path2 = '../dataset/train/pos.txt'

# 3000条测试集
# findPath1 = '../dataset/test/test.txt_utf8'
################################################

# 1000条微博测试集
# findPath1 = '../dataset/WeiBoYuLiao/test.txt'

# 1000条淘宝测试集
findPath1 = '../dataset/TaoBaoPingLun/test.txt'


# 训练分词
def train_fenci():
    list_words = []

    test_text = open(file_path1, 'r', encoding='utf-8').readlines()
    for line in test_text:
        # 清洗数据
        text = ''.join(line.split())
        # 实现目标文本中对正则表达式中的模式字符串进行替换
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+", "", text)
        # 利用jieba包自动对处理后的文本进行分词
        test_list = jieba.cut(text, cut_all=False)
        # 得到所有分解后的词
        list_words.append(' '.join(test_list))

    test_text = open(file_path2, 'r', encoding='utf-8').readlines()
    for line in test_text:
        # 清洗数据
        text = ''.join(line.split())
        # 实现目标文本中对正则表达式中的模式字符串进行替换
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+", "", text)
        # 利用jieba包自动对处理后的文本进行分词
        test_list = jieba.cut(text, cut_all=False)
        # 得到所有分解后的词
        list_words.append(' '.join(test_list))
    return list_words


# 测试分词
def test_fenci():
    neg_words = []

    lines = open(findPath1, 'r', encoding='utf-8').readlines()
    for line in lines:
        temp = ''.join(line.split())
        # 实现目标文本中对正则表达式中的模式字符串进行替换
        temp = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+", "", temp)
        # 利用jieba包自动对处理后的文本进行分词
        temp_list = jieba.cut(temp, cut_all=False)
        # 得到所有分解后的词
        neg_words.append(' '.join(temp_list))
    return neg_words


def get_content(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = []
        for l in f.readlines():
            l = l.strip().replace(u'\u3000', u'')
            content.append(l)
    return content


if __name__ == '__main__':
    tfidf_vect = TfidfVectorizer(analyzer='word', stop_words=['我', '你', '是', '的', '在', '这里'])
    train_tfidf = tfidf_vect.fit_transform(train_fenci())
    test_tfidf = tfidf_vect.transform(test_fenci())
    # words = tfidf_vect.get_feature_names()
    # print(words)
    # print(train_tfidf)
    # print(len(words))
    # print(train_tfidf)
    # print(tfidf_vect.vocabulary_)

    # SGD模型
    # lr = SGDClassifier(loss='log', penalty='l1')

    # SVM模型
    lr = SVC(kernel='rbf', verbose=True)

    # NB
    # lr = MultinomialNB()

    # ANN
    # lr = MLPClassifier(hidden_layer_sizes=1, activation='logistic', solver='lbfgs', random_state=0)

    # LR
    # lr = LogisticRegression(C=1, penalty='l2')

    # 训练
    lr.fit(train_tfidf, ['neg'] * len(open(file_path1, 'r', encoding='utf-8').readlines()) +
           ['pos'] * len(open(file_path2, 'r', encoding='utf-8').readlines()))

    # 预测
    y_pred = lr.predict(test_tfidf)
    print(y_pred)

    # 情感词典模块
    # 为测试集加入标签
    comment = get_content(findPath1)
    pos_lable = [1 for i in range(1500)]
    neg_lable = [-1 for i in range(1500)]
    lables = pos_lable + neg_lable

    # 情感词典判定语句
    from es import sentiment

    predictions = []
    for line in comment:
        predictions.append(sentiment(line))

    # 统计结果和准确率
    sum_counter = 0
    pos_right = 0
    pos_wrong = 0
    neg_right = 0
    neg_wrong = 0
    j = 0
    for i in y_pred:
        x = predictions[j]
        if sum_counter < 500:
            if i == 'pos' or x > 0:
                pos_right += 1
            else:
                pos_wrong += 1
        else:
            if i == 'neg' or x < 0:
                neg_right += 1
            else:
                neg_wrong += 1
        sum_counter += 1
        j += 1

    # precision
    P = pos_right / (pos_right + pos_wrong)
    # recall
    R = pos_right / (pos_right + neg_wrong)
    # f-score
    F = 2 * pos_right / (2 * pos_right + pos_wrong + neg_wrong)
    right = pos_right + neg_right
    wrong = pos_wrong + neg_wrong
    percent = right / (right + wrong)
    print("判断正确:", right)
    print("判断错误:", wrong)
    print("正确率：", percent)
    print("精准率：", P)
    print("召回率：", R)
    print("f值：", F)
