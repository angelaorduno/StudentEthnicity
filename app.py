from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
filename = 'LogisticRegression.pkl'
model = pickle.load(open(filename, 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the web form
    math = int(request.form['math score'])
    reading = int(request.form['reading score'])
    writing = int(request.form['writing score'])

    # Create a data frame with the user input
    data = pd.DataFrame([[math, reading, writing]], columns=['math score', 'reading score', 'writing score'])

    # Make a prediction using the model
    prediction = model.predict(data)

    # Render a new web page with the prediction
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)