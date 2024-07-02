from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('sentiment_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    prediction = model.predict([message])[0]
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
