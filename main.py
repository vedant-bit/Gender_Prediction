import numpy as np
import os
import numpy as np
from keras.models import load_model
from tensorflow.keras import backend
from flask_bootstrap import Bootstrap
import pickle
from flask import Flask,render_template, request


app = Flask(__name__)
Bootstrap(app)


model = load_model('gender_model.h5')
CI = pickle.load(open('transform.pkl' , 'rb'))

def set_flag(i):
    tmp = np.zeros(28);
    tmp[i] = 1
    return(tmp)


def tr(naam,):
    maxlen = 20
    for j in str(naam):
        l = CI[j]
        tmp = [set_flag(l)]
    for k in range(0,maxlen - len(str(naam))):
        tmp.append(set_flag(CI["END"]))
    X =[]
    X.append(tmp)
    return np.asarray(X)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("Home.html")


@app.route('/predict', methods=['POST', "GET"])
def predict():
    if request.method == 'POST':
        namequery = request.form['namequery']
        data = namequery
        vect = tr(data)
        my_prediction = model.predict(vect)

    return render_template('result.html',prediction = my_prediction,name = namequery.upper())

if __name__ == '__main__':
	app.run(debug=True)

