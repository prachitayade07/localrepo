from flask import Flask, request, jsonify, render_template
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load pre-trained models
kmeans_model = joblib.load("models/kmeans_url_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
spam_model = joblib.load("models/spam_detection_model.pkl")
spam_vectorizer = joblib.load("models/spam_vectorizer.pkl")  # for spam TF-IDF vectorization

# Safe domain whitelist
safe_domains = [
    "google.com", "openai.com", "github.com", "wikipedia.org", "microsoft.com",
    "apple.com", "amazon.com", "stackoverflow.com", "youtube.com", "linkedin.com", "sbi.com"
]

# URL pattern for validation
url_pattern = re.compile(
    r'^(https?:\/\/)?'              # Optional http/https
    r'(([A-Za-z0-9-]+\.)+[A-Za-z]{2,6})'  # Domain name
    r'(\/[^\s]*)?$'                 # Optional path
)

def is_valid_url(url):
    """Validate URL format."""
    return re.match(url_pattern, url) is not None

def is_whitelisted(url):
    """Check if URL domain is in the safe list."""
    return any(domain in url for domain in safe_domains)

def extract_url_features(url):
    """Vectorize the URL using TF-IDF."""
    return vectorizer.transform([url])

# Custom mapping based on model analysis (adjust if needed)
cluster_to_category = {
    0: 'phishing',
    1: 'defacement',
    2: 'benign',
    3: 'malicious'
}

def classify_url(url):
    """Classify the given URL."""
    if is_whitelisted(url):
        return {'url': url, 'classification': 'benign', 'score': 99.99, 'phishing_score': 0, 'defacement_score': 0}

    features = extract_url_features(url)
    cluster = kmeans_model.predict(features)[0]

    classification = cluster_to_category.get(cluster, 'unknown')

    # Calculate distances for scoring
    distances = kmeans_model.transform(features)
    min_distance = np.min(distances)
    score = 100 - (min_distance * 10)
    score = max(0, min(99, score))

    # Here, assume Phishing, Defacement, and Malicious scores are calculated
    phishing_score = 0
    defacement_score = 0
    malicious_score = 0

    if classification == 'phishing':
        phishing_score = round(score, 2)
    elif classification == 'defacement':
        defacement_score = round(score, 2)
    elif classification == 'malicious':
        malicious_score = round(score, 2)

    return {
        'url': url,
        'classification': classification,
        'score': round(score, 2),
        'phishing_score': phishing_score,
        'defacement_score': defacement_score,
        'malicious_score': malicious_score
    }

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/detect_url', methods=['POST'])
def detect_url():
    data = request.json
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    if not is_valid_url(url):
        return jsonify({"error": "Invalid URL format. Please enter a valid URL."}), 400

    result = classify_url(url)

    return jsonify(result)

def classify_spam(text):
    """Classifies text as spam or not spam using a trained model."""
    # Basic input sanity check
    if len(text) < 5 or not re.search(r'[a-zA-Z0-9]', text):
        return {'error': 'Input text is too short or invalid.'}

    # Check for gibberish or meaningless input
    words = text.strip().split()
    real_words = [word for word in words if re.search(r'[a-zA-Z0-9]', word)]

    if len(real_words) < 2:
        return {'error': 'Input text must contain at least two meaningful words.'}

    # Vectorize and predict
    transformed = spam_vectorizer.transform([text])
    spam_score = spam_model.predict_proba(transformed)[0][1] * 100
    return {'text': text, 'spam_score': round(spam_score, 2)}

@app.route('/detect_spam', methods=['POST'])
def detect_spam():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = classify_spam(text)

    if 'error' in result:
        return jsonify({"error": result['error']}), 400

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
