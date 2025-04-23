import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("emotions.csv")
df.dropna(inplace=True)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=2000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LinearSVC(C=1.0, max_iter=5000)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "model/svm_model.joblib")
joblib.dump(vectorizer, "model/vectorizer.joblib")

# Evaluate
print("Accuracy:", accuracy_score(y_test, model.predict(X_test_vec)))
