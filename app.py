from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv('email.csv', encoding='latin-1')

x = np.array(data["Message"])
y = np.array(data["Category"])
cv = CountVectorizer()
X = cv.fit_transform(x)  # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sample = request.form['message']
        data = cv.transform([sample]).toarray()
        prediction = clf.predict(data)[0]
        return render_template('index.html', prediction=prediction)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)