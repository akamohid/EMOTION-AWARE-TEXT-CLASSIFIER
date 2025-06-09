# Utils
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_vader_score(text):
    score = analyzer.polarity_scores(text)
    return score['compound']
