# sentiment_train.py
import os
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 可选停用词
def load_stopwords(path='stopwords.txt'):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return set([w.strip() for w in f if w.strip()])
    return set()

stopwords = load_stopwords()

def jieba_tokenizer(text):
    tokens = jieba.lcut(str(text))
    if not stopwords:
        return [t for t in tokens if t.strip()]
    return [t for t in tokens if t.strip() and t not in stopwords]

def load_data(path='data/train.csv'):
    df = pd.read_csv(path, encoding='utf-8')
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV 文件必须包含 'text' 和 'label' 两列")
    df = df.dropna(subset=['text','label'])
    df['label'] = df['label'].astype(int)
    return df

def main():
    os.makedirs('model', exist_ok=True)
    df = load_data('data/train.csv')
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(df) > 1 else None
    )

    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer, ngram_range=(1,2), max_features=20000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    print("准确率 (accuracy):", accuracy_score(y_test, y_pred))
    print("分类报告：")
    print(classification_report(y_test, y_pred, digits=4))
    print("混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))

    # 保存模型和向量器
    joblib.dump(vectorizer, 'model/vectorizer.joblib')
    joblib.dump(clf, 'model/clf.joblib')
    print("✅ 模型已保存到 model/vectorizer.joblib 和 model/clf.joblib")

if __name__ == '__main__':
    main()
