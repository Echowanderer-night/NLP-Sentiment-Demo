from flask import Flask, request, render_template_string
import joblib
import jieba
import os

# 加载模型和向量器
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

vectorizer = joblib.load('model/vectorizer.joblib')
clf = joblib.load('model/clf.joblib')

# Flask 应用
app = Flask(__name__)

HTML = """
<!doctype html>
<title>中文情感分析小 Demo</title>
<h2>中文情感分析（正/负）</h2>
<form method=post>
  <textarea name=text rows=4 cols=60 placeholder="在这里输入一句中文..."></textarea><br>
  <input type=submit value="预测">
</form>
{% if result %}
  <h3>结果：{{ result }}</h3>
{% endif %}
"""

@app.route('/', methods=['GET','POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form.get('text', '')
        if text.strip():
            X = [text]
            X_tfidf = vectorizer.transform(X)
            pred = clf.predict(X_tfidf)[0]
            proba = clf.predict_proba(X_tfidf).max()
            result = f"{'正面' if pred==1 else '负面'} （置信度 {proba:.2f}）"
    return render_template_string(HTML, result=result)

if __name__ == '__main__':
    app.run(debug=True)
