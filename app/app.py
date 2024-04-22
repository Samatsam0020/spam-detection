from flask import Flask, request, render_template
import pickle
import numpy as np

# Load model et preprocessing functions
with open('model_pickle', 'rb') as f:
    model = pickle.load(f)

with open('vocab.pickle', 'rb') as f:
    vocab = pickle.load(f)


def word_to_vec(phrase, vocab=vocab):
    vec = np.zeros(len(vocab))
    word = phrase.split()

    for w in word:
        if w in vocab:
            indice = vocab.index(w)
            vec[indice] += 1
    return np.array([vec])


# Flask Application

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict_spam():
    if request.method == 'POST':
        email = request.form['email']

        vecteur = word_to_vec(email)
        prediction = model.predict(vecteur)

        if prediction == 1:
            result = "It's spam."
        else:
            result = "It's not a spam."

        return render_template('index.html', result=result)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
