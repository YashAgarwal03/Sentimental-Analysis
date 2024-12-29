from flask import Flask, render_template, request
from deployment.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.form['text_input']
        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.predict(user_input)

        sentiment = "Positive" if prediction > 0 else "Negative"
        emoji = "happy.jpg" if prediction > 0 else "sad.jpg"
        return render_template('result.html', user_input=user_input, sentiment=sentiment, emoji=emoji)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
