# Spam SMS Detection

## Project Overview
This project is a Machine Learning model that classifies SMS messages as Spam or Ham (Not Spam).

## Features
- Detects spam SMS messages
- Uses TF-IDF for text vectorization
- Uses Multinomial Naive Bayes for classification
- Achieved 96.23% accuracy

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Dataset
SMS Spam Collection Dataset

## How to Run

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Run the project:

```bash
python3 spam_sms_detection.py
```

## Example

Input:

```
Congratulations! You have won a FREE iPhone.
```

Output:

```
Prediction: Spam
```

Input:

```
Hi Aman, let's meet tomorrow.
```

Output:

```
Prediction: Ham (Not Spam)
```

## Accuracy

Model Accuracy: **96.23%**