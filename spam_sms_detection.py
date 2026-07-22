import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Convert text into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])

# Labels
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Take input from user
message = input("Enter an SMS: ")

# Convert to TF-IDF
message_vector = vectorizer.transform([message])

# Predict
prediction = model.predict(message_vector)

# Show result
if prediction[0] == 1:
    print("Prediction: Spam")
else:
    print("Prediction: Ham (Not Spam)")