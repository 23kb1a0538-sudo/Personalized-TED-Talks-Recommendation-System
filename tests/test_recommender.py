"""
Unit and Integration Tests for TED Talks Recommendation System
"""

import unittest
import numpy as np
import pandas as pd
from src.models import HybridTEDRecommender, ContentBasedRecommender, CollaborativeFilteringRecommender


class TestTEDRecommender(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = HybridTEDRecommender()
        cls.df = cls.engine.df

    def test_dataset_loaded(self):
        self.assertGreater(len(self.df), 2000)
        self.assertIn('title', self.df.columns)
        self.assertIn('main_speaker', self.df.columns)
        self.assertIn('content_soup', self.df.columns)

    def test_content_recommendations(self):
        # Talk 0: Ken Robinson - Do schools kill creativity?
        recs = self.engine.recommend(talk_id=0, top_n=5)
        self.assertEqual(len(recs), 5)
        self.assertNotEqual(recs[0]['talk_id'], 0)  # Should not recommend self
        for r in recs:
            self.assertIn('title', r)
            self.assertIn('similarity_score', r)
            self.assertGreater(r['similarity_score'], 0.0)

    def test_collaborative_recommendations(self):
        recs = self.engine.recommend(user_id=1, top_n=5)
        self.assertEqual(len(recs), 5)
        for r in recs:
            self.assertIn('title', r)
            self.assertIn('predicted_rating', r)

    def test_hybrid_recommendations(self):
        recs = self.engine.recommend(talk_id=0, user_id=1, alpha=0.6, top_n=5)
        self.assertEqual(len(recs), 5)
        for r in recs:
            self.assertIn('hybrid_score', r)
            self.assertIn('content_component', r)
            self.assertIn('collab_component', r)

    def test_query_search(self):
        recs = self.engine.recommend(query="machine learning artificial intelligence", top_n=3)
        self.assertEqual(len(recs), 3)
        for r in recs:
            self.assertIn('title', r)
            self.assertIn('similarity_score', r)

    def test_cold_start_fallback(self):
        # User ID out of bounds should gracefully fallback to popular talks without crashing
        recs = self.engine.recommend(user_id=99999, top_n=3)
        self.assertEqual(len(recs), 3)


if __name__ == '__main__':
    unittest.main()
