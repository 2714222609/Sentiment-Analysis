import jieba
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# 20000条训练集
from sklearn.tree import DecisionTreeClassifier

neg = pd.read_excel('neg.xls', header=None)
pos = pd.read_excel('pos.xls', header=None)

neg['words'] = neg[0].apply(lambda x: jieba.lcut(x))
pos['words'] = pos[0].apply(lambda x: jieba.lcut(x))
x = np.concatenate((pos['words'], neg['words']))
y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
w2v = Word2Vec(size=300, min_count=10)
w2v.build_vocab(x)
w2v.train(x, total_examples=w2v.corpus_count, epochs=10)
w2v.save(u'w2v_model.model')
print("w2v_model saved")


def total_vec(words):
    vec = np.zeros(300).reshape((1, 300))
    for word in words:
        try:
            vec += w2v.wv[word].reshape((1, 300))
        except KeyError:
            continue
    return vec


train_vec = np.concatenate([total_vec(words) for words in x])

# SGD模型
# model = SGDClassifier(loss='log', penalty='l1')

# SVM模型
model = SVC(kernel='rbf', verbose=True)

# NB
# model = BernoulliNB()

# ANN
# model = MLPClassifier(hidden_layer_sizes=1, activation='logistic', solver='lbfgs', random_state=0)

# LR
# model = LogisticRegression(C=1, penalty='l2')

# DT（决策树）
# lr = DecisionTreeClassifier()

# RF（随机森林）
# lr = RandomForestClassifier(n_estimators=28)

# AdaBoost
# lr = AdaBoostClassifier()

# GBM（梯度提升）
lr = GradientBoostingClassifier()

# 训练
model.fit(train_vec, y)
joblib.dump(model, 'model.pkl')
print("machine learning model saved")


def total_vec(words):
    w2v = Word2Vec.load('w2v_model.model')
    vec = np.zeros(300).reshape((1, 300))
    for word in words:
        try:
            vec += w2v.wv[word].reshape((1, 300))
        except KeyError:
            continue
    return vec


def svm_predict():
    # 3000条测试集
    df = pd.read_csv("test.csv")

    model = joblib.load('model.pkl')
    comment_sentiment = []
    sum_counter = 0
    pos_right = 0
    pos_wrong = 0
    neg_right = 0
    neg_wrong = 0
    for string in df['内容']:
        words = jieba.lcut(str(string))
        words_vec = total_vec(words)
        result = model.predict(words_vec)
        comment_sentiment.append('积极' if int(result[0]) else '消极')

        if sum_counter < 1500:
            if int(result[0]) == 1:
                pos_right += 1
            else:
                pos_wrong += 1
        else:
            if int(result[0]) == 0:
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


svm_predict()
