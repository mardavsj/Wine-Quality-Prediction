from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
     if request.method == 'POST':
            # Get input values from the form
        features = [float(request.form['fixed_acidity']),
                    float(request.form['volatile_acidity']),
                    float(request.form['citric_acid']),
                    float(request.form['residual_sugar']),
                    float(request.form['chlorides']),
                    float(request.form['free_sulfur_dioxide']),
                    float(request.form['total_sulfur_dioxide']),
                    float(request.form['density']),
                    float(request.form['pH']),
                    float(request.form['sulphates']),
                    float(request.form['alcohol'])]

        # Create a DataFrame with the input values
        data = pd.DataFrame([features], columns=['fixed_acidity', 'volatile_acidity', 'citric_acid',
                                                 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                                                 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'])

        # Make prediction
        prediction = model.predict(data)

        # Interpret the predicted quality score
        if prediction == 1:
            message = "The predicted wine quality is bad."
        elif prediction == 10:
            message = "The predicted wine quality is excellent."
        else:
            message = "The predicted wine quality is moderate."

        return render_template('result.html', prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True)
