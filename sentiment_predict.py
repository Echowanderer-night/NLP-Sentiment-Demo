import jieba
import joblib
import os

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

# 加载模型
vectorizer = joblib.load('model/vectorizer.joblib')
clf = joblib.load('model/clf.joblib')

def predict(text):
    X = [text]
    X_tfidf = vectorizer.transform(X)
    pred = clf.predict(X_tfidf)[0]
    proba = clf.predict_proba(X_tfidf).max()
    return int(pred), float(proba)

if __name__ == '__main__':
    print("输入一句中文，回车得到情感预测（输入 exit 退出）")
    while True:
        s = input(">>> ")
        if s.strip().lower() in ('exit','quit'):
            break
        if not s.strip():
            continue
        label, conf = predict(s)
        print(f"预测：{'正面' if label==1 else '负面'}，置信度：{conf:.2f}")
