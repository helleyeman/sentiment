from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = str(data.get("text", ""))
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return jsonify({"sentiment": pred})

if __name__ == "__main__":
    app.run(debug = True, port = 8000)