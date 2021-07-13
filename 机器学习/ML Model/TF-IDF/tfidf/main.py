from typing import List

from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
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


from sklearn.tree import DecisionTreeClassifier
# 20000条训练集
# file_path1 = '../dataset/train/neg.txt'
# file_path2 = '../dataset/train/pos.txt'

# 10000条微博语录训练集
file_path1 = '../dataset/WeiBoYuLiao/neg_train.txt'
file_path2 = '../dataset/WeiBoYuLiao/pos_train.txt'


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
    # 3000条测试集
    findPath1 = '../dataset/test/test.txt_utf8'

    # 500条测试集
    # findPath1 = '../dataset/WeiBoYuLiao/test.txt'
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

    # SGD（随机梯度下降）
    # lr = SGDClassifier(loss='log', penalty='l1')

    # SVM（支持向量机）
    lr = SVC(kernel='rbf', verbose=True)

    # NB（朴素贝叶斯）
    # lr = MultinomialNB()

    # ANN（人工神经网络）
    # lr = MLPClassifier(hidden_layer_sizes=1, activation='logistic', solver='lbfgs', random_state=0)

    # LR（逻辑回归）
    # lr = LogisticRegression(C=1, penalty='l2')

    # DT（决策树）
    # lr = DecisionTreeClassifier()

    # RF（随机森林）
    # lr = RandomForestClassifier(n_estimators=28)

    # AdaBoost
    # lr = AdaBoostClassifier()

    # GBM（梯度提升）
    # lr = GradientBoostingClassifier()

    # 训练
    lr.fit(train_tfidf, ['neg'] * len(open(file_path1, 'r', encoding='utf-8').readlines()) +
           ['pos'] * len(open(file_path2, 'r', encoding='utf-8').readlines()))

    # 预测
    y_pred = lr.predict(test_tfidf)
    print(y_pred)

    # 统计结果和准确率
    sum_counter = 0
    pos_right = 0
    pos_wrong = 0
    neg_right = 0
    neg_wrong = 0
    for i in y_pred:
        if sum_counter < 1500:
            if i == 'pos':
                pos_right += 1
            else:
                pos_wrong += 1
        else:
            if i == 'neg':
                neg_right += 1
            else:
                neg_wrong += 1
        sum_counter += 1

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
