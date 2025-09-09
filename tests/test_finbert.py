import os
import types
import unittest


class TestFinBertSentiment(unittest.TestCase):
    def setUp(self):
        # Ensure module can be imported
        from src.nlp import sentiment_analysis as sa  # noqa: F401

    def test_score_sentiment_basic(self):
        from src.nlp import sentiment_analysis as sa
        self.assertIsInstance(sa.score_sentiment(""), float)
        self.assertEqual(sa.score_sentiment(None), 0.0)  # type: ignore[arg-type]

        pos = sa.score_sentiment("Strong earnings beat and positive guidance.")
        neg = sa.score_sentiment("Massive losses and bankruptcy risk.")
        self.assertGreaterEqual(pos, -1.0)
        self.assertLessEqual(pos, 1.0)
        self.assertGreaterEqual(neg, -1.0)
        self.assertLessEqual(neg, 1.0)

    def test_batch_api_shape(self):
        from src.nlp import sentiment_analysis as sa
        texts = [
            "Revenue up 20% QoQ; margins expanding.",
            "SEC probe and liquidity crunch.",
            "",
        ]
        out = sa.analyze_sentiments(texts)
        self.assertEqual(len(out), len(texts))
        for item in out:
            self.assertIn(item["sentiment"], ["positive", "negative", "neutral"])  # type: ignore[index]
            self.assertTrue(-1.0 <= float(item["polarity"]) <= 1.0)  # type: ignore[index]
            self.assertIsInstance(item["confidence"], float)  # type: ignore[index]
            self.assertIn(item["source"], ["finbert", "aws_comprehend", "unavailable"])  # type: ignore[index]

    def test_fallback_mock(self):
        # Simulate FinBERT failure by monkeypatching initializer
        from src.nlp import sentiment_analysis as sa

        original_init = sa._init_finbert_pipeline
        sa._FINBERT_PIPELINE = None
        sa._init_finbert_pipeline = lambda: (_ for _ in ()).throw(RuntimeError("fail"))  # type: ignore[assignment]
        try:
            out = sa.analyze_sentiments(["Positive sales outlook."])
            self.assertEqual(len(out), 1)
            self.assertIn(out[0]["source"], ["aws_comprehend", "unavailable"])  # type: ignore[index]
        finally:
            sa._init_finbert_pipeline = original_init  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()

