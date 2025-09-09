from textblob import TextBlob

def score_sentiment(text):
    if not text or not isinstance(text, str):
        return 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity
