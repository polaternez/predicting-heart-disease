import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app = Flask(__name__)

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predictdata():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        # Unbox data from request form
        data = CustomData(
            Age=float(request.form.get('Age')),
            Sex=request.form.get('Sex'),
            ChestPainType=request.form.get('ChestPainType'),
            RestingBP=float(request.form.get('RestingBP')),
            Cholesterol=float(request.form.get('Cholesterol')),
            FastingBS=float(request.form.get('FastingBS')),
            RestingECG=request.form.get('RestingECG'),
            MaxHR=float(request.form.get('MaxHR')),
            ExerciseAngina=request.form.get('ExerciseAngina'),
            Oldpeak=float(request.form.get('Oldpeak')),
            ST_Slope=request.form.get('ST_Slope'),
        )

        data_df = data.get_data_as_dataframe()
        print(data_df)
        
        print("Before Prediction")

        predict_pipeline = PredictPipeline()

        print("Mid Prediction")

        prediction = predict_pipeline.predict(data_df)

        print("After Prediction")

        output = ""
        if prediction[0] == 0:
            output = "Patient looks healthy!!"
        elif prediction[0] == 1:
            output = "Patient probably has heart disease!!"

        return render_template('home.html', results=output)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)        