环境说明：

python 3.x

文件夹说明：

ML Model：
该文件夹为只使用机器学习进行文本分析的代码，其中按照向量表示方法的不同分为
TF-IDF和WordVec。TF-IDF文件夹中核心代码在tfidf，数据集在dataset中。
WordVec中核心代码为main.py，训练集为neg和pos，测试集为test。

ML Model and emotion_dic：
该文件夹为结合机器学习和情感词典进行文本分析的代码，同样的按照向量表示方法的不同分为
TF-IDF和WordVec。与ML Model相比只增加了一个emotion_dict文件夹，此文件夹用来保存情感词典。

代码使用说明：

无论是TF-IDF还是和WordVec，都从核心代码main.py入手，代码注释中注释了不同的数据集和模型，使用者
只需要改动注释就可以使用不同的模型和数据集，而后直接编译运行即可，模型的构建需要一些时间，请耐心等待结果输出

数据集使用说明：
由于数据集过大，导致无法提交，故将文件夹保存到百度网盘中，请使用者自行下载。
机器学习数据集tf-dataset保存到文本情感分析代码\机器学习\ML Model\TF-IDF中
tf-e-dataset保存到文本情感分析代码\机器学习\ML Model and emotion_dic\TF-IDF中
w-e-data保存到文本情感分析代码\机器学习\ML Model and emotion_dic\Word2Vec和
文本情感分析代码\机器学习\ML Model\Word2Vec中
链接：https://pan.baidu.com/s/1lHbte6elesVyEMvOGzLMJg 
提取码：tsxd