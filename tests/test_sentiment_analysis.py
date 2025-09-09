# tests/test_sentiment_analysis.py
import sys
import os

# Dynamically add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from nlp.sentiment_analysis import score_sentiment


def test_positive_sentiment():
    text = "This is a great opportunity with a lot of potential!"
    score = score_sentiment(text)
    assert score > 0

def test_negative_sentiment():
    text = "This is a terrible experience and I regret everything."
    score = score_sentiment(text)
    assert score < 0

def test_neutral_sentiment():
    text = "The company held a press conference."
    score = score_sentiment(text)
    assert -0.1 <= score <= 0.1

def test_empty_string():
    text = ""
    score = score_sentiment(text)
    assert score == 0.0

def test_non_string_input():
    score = score_sentiment(None)
    assert score == 0.0
