import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
DATA_PATH = "../dataset/spam.csv"
df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
 
# Print columns to verify structure
print("Columns before cleaning:", df.columns)

# Keep only relevant columns
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

# Convert labels to binary (1 = spam, 0 = not spam)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Drop any NaN values
df = df.dropna()

# Print the first few rows to verify cleaning
print("Cleaned dataset preview:\n", df.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Spam detection model accuracy: {accuracy:.2f}")

# Save model and vectorizer
joblib.dump(model, "../models/spam_detection_model.pkl")
joblib.dump(vectorizer, "../models/spam_vectorizer.pkl")

print("✅ Spam detection model and vectorizer saved successfully!")
