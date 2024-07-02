# senti-analysis-flask

structure 

sentiment_analysis_project/
│
├── app.py
├── data_preprocessing_and_model_training.py
├── sentiment_model.pkl
└── templates/
    ├── index.html
    └── result.html


Sentiment Analysis Flask Application
This project implements a sentiment analysis web application using Flask. It allows users to input text and get a prediction of whether the sentiment is positive or negative based on a pre-trained model.

Setup
Prerequisites
Python 3.x installed on your system
Git installed on your system (optional, for cloning the repository)
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/machphy/senti-analysis-flask.git
cd senti-analysis-flask
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Data Preprocessing and Model Training
Before running the application, the sentiment analysis model (sentiment_model.pkl) needs to be trained. Follow these steps:

bash
Copy code
python data_preprocessing_and_model_training.py
This script downloads the movie reviews dataset, preprocesses the data, trains a TF-IDF vectorizer and a Multinomial Naive Bayes classifier, evaluates the model's accuracy, and saves it as sentiment_model.pkl.

Running the Application
To start the Flask application locally:

bash
Copy code
python app.py
The application will run on http://127.0.0.1:5000/ (or http://localhost:5000/). Open this URL in your web browser to access the application.

Usage
Enter text into the provided text area on the homepage.
Click the "Analyze" button to submit the text for sentiment analysis.
View the predicted sentiment result on the result page.
Files and Directories
app.py: Flask application script handling web routes and model integration.
data_preprocessing_and_model_training.py: Script for data preprocessing, model training, and saving the trained model.
sentiment_model.pkl: Pre-trained sentiment analysis model.
templates/: Directory containing HTML templates (index.html, result.html) for the web interface.
Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

License
MIT

