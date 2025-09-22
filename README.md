中文文本情感分析小系统（基础版）
这是一个从零开始搭建的"中文文本情感分析系统"，包含数据处理、模型训练、命令行预测以及网页 Demo。  
项目目标：学习 NLP（自然语言处理）的基本流程，并展示从建模到简单应用的完整实现。  

项目结构
NLPdemo1/
│── data/ # 示例数据
│ └── train.csv
│── model/ # 训练好的模型（vectorizer.joblib, clf.joblib）
│── sentiment_train.py # 模型训练脚本
│── sentiment_predict.py # 命令行预测
│── app.py # Flask 网页 Demo
│── requirements.txt # 依赖库清单
│── README.md # 项目说明文档

环境依赖
推荐使用虚拟环境（venv），Python 版本 >= 3.9  

安装依赖：
bash
pip install -r requirements.txt

主要依赖库：
jieba
pandas / numpy
scikit-learn
joblib
flask

使用方法
1. 数据准备
运行data/make_sample.py（会生成一个示例训练数据集 data/train.csv）：
python data/make_sample.py

2. 模型训练
训练模型并保存：
python sentiment_train.py

3. 命令行预测
运行交互式预测脚本：
python sentiment_predict.py
示例：
![交互 example](image/text_example.png)

4. 网页 Demo
运行 Flask 应用：
python app.py
浏览器打开： http://127.0.0.1:5000
示例效果：
![网页 Demo](image/web_demo.png)

模型介绍
文本表示：TF-IDF 向量
分类模型：逻辑回归（Logistic Regression）
评估指标：准确率、精确率、召回率、F1 值
在小数据集上能有效区分简单的正面/负面情感。

学到的内容
通过本项目，我掌握了：
中文文本分词（jieba）
特征提取（TF-IDF）
机器学习建模（Logistic Regression）
模型保存与加载（joblib）
命令行交互与网页部署（Flask）

下一步计划
尝试 Sentence-BERT 句向量提升效果
使用更大规模的公开中文数据集
尝试细粒度情感分类（如：喜悦、愤怒、悲伤）
将系统部署到云端（例如 Hugging Face Space / Vercel / 阿里云）
