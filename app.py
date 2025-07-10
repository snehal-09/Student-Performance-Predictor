from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('student_perf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours_studied = float(request.form['hours'])
        prev_scores = float(request.form['previous'])
        activity = request.form['activity']  # 'Yes' or 'No'
        sleep = float(request.form['sleep'])
        papers = float(request.form['papers'])

        # Convert categorical to numeric: Yes=1, No=0
        activity_value = 1 if activity.lower() == 'yes' else 0

        # Create input for prediction
        input_features = np.array([[hours_studied, prev_scores, activity_value, sleep, papers]])
        prediction = model.predict(input_features)[0]

        return render_template('index.html',
                               prediction_text=f"📊 Predicted Performance Index: {round(prediction, 2)}")
    except Exception as e:
        return render_template('index.html',
                               prediction_text="❌ Error: Please enter valid inputs.")

if __name__ == '__main__':
    app.run(debug=True)
