from flask import Flask, request, render_template
from joblib import load
import os

app = Flask(__name__)

# Load models from model/ folder
model_path = os.path.join('model', 'model.joblib')
vectorizer_path = os.path.join('model', 'vectorizer.joblib')
model = load(model_path)
vectorizer = load(vectorizer_path)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        result = "Toxic" if prediction == 1 else "Safe"
        return render_template('result.html', text=text, result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)