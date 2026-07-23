import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv(
    "train_data.txt",
    sep=" ::: ",
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"],
    engine="python"
)

print(df.shape)

print(df.isnull().sum())

vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(df["DESCRIPTION"])

print(X.shape)

y = df["GENRE"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

movie_description = input("Enter movie description: ")

movie_vector = vectorizer.transform([movie_description])

prediction = model.predict(movie_vector)

print("Predicted Genre:", prediction[0])