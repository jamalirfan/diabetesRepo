import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('DiabetesPredictor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    preg = request.form['preg']
    glucose = request.form['glucose']
    bp = request.form['bp']
    skin = request.form['skin']
    insulin = request.form['insulin']
    bmi = float(request.form['BMI'])
    diaPedFun = float(request.form['DiabetesPedigreeFunction'])
    age = float(request.form['Age']) 
    
    finalFeatures = np.array([[preg,glucose,bp,skin,insulin,bmi,diaPedFun,age]])
    prediction = model.predict(finalFeatures)

    

    return render_template('index.html', prediction_text='Outcome is {}'.format(round(prediction[0])))


if __name__ == "__main__":
    app.run(debug=True)