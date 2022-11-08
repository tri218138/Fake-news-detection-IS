from flask import Flask, render_template, request, jsonify
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
import numpy as np

app = Flask(__name__)
ps = PorterStemmer()

model = pickle.load(open('Model/logreg_fakenews.pickle', 'rb'))
tfidfvect = pickle.load(open('Model/tfidf.pickle', 'rb'))


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


def predict(text, title, author):
    content = title + ' ' + author + ' ' + text
    tokens = re.sub('[^a-zA-Z]', ' ', content)
    review = tokens.lower()
    review = review.split()
    review = [ps.stem(word)
              for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)

    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'giả' if model.predict(review_vect)[0] else 'thật'
    print(model.predict(review_vect))

    diction = measure_importance(review_vect)
    # print(diction)
    return prediction, diction, tokens


@app.route('/', methods=['POST'])
def webapp():
    text = request.form['textInput']
    author = request.form['authorInput']
    title = request.form['titleInput']
    prediction, weighted_stems, tokens = predict(text, title, author)

    coloring = dict()
    for word in tokens.split():
        stemmed = ps.stem(word.lower()) if not word in stopwords.words('english') else word.lower()
        if stemmed in weighted_stems:
            coloring[word] = weighted_stems[stemmed] if stemmed in weighted_stems else 0

    return render_template('index.html', original=text, textInput=text, result=prediction, dict=coloring)


@app.route('/predict/', methods=['GET', 'POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)


def measure_importance(review_vect):
    word_indexes = np.argwhere(review_vect[0])
    word_values = review_vect[0, word_indexes][:, 0]
    # print(word_values)
    # print(word_indexes[:, 0])
    word_coeffs = model.coef_[0][word_indexes[:, 0]]
    word_importance = word_coeffs * word_values
    # print(word_importance)
    return dict(zip(tfidfvect.get_feature_names_out()[word_indexes][:, 0], word_importance))


if __name__ == "__main__":
    app.run(debug=True)
