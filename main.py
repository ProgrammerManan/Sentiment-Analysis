import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
from data import finance_data, healthcare_data, social_media_data, customer_reviews

def analyze_sentiment(data, model):
    """Analyzes sentiment using the specified model."""
    results = []
    for item in data:
        text = item['text']
        actual_sentiment = item['sentiment']

        if model == 'textblob':
            sentiment = TextBlob(text).sentiment.polarity
        elif model == 'vader':
            sentiment = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
        else:
            raise ValueError("Invalid model specified")

        predicted_sentiment = 'positive' if sentiment >= 0.2 else 'negative'
        results.append({'text': text, 'actual_sentiment': actual_sentiment, 'predicted_sentiment': predicted_sentiment})

    return pd.DataFrame(results)

def evaluate_model(df):
    """Evaluates the model's performance."""
    report = classification_report(df['actual_sentiment'], df['predicted_sentiment'], output_dict=True)
    accuracy = report['accuracy'] * 100
    return accuracy, report

def compare_models(datasets, models):
    """Compares multiple models on multiple datasets."""
    for dataset_name, data in datasets.items():
        print(f"\nDataset: {dataset_name}")
        for model in models:
            results = analyze_sentiment(data, model)
            accuracy, report = evaluate_model(results)
            print(f"Model: {model} - Accuracy: {accuracy:.2f}%")
        print("\n" + "-"*50)


datasets = {
    'finance': finance_data,
    'healthcare': healthcare_data,
    'social_media': social_media_data,
    'customer_reviews': customer_reviews
}

# Define models to compare
models = ['textblob', 'vader']

# Run the comparison
compare_models(datasets, models)
