import unittest
import joblib
from sklearn.cluster import KMeans

class TestClusteringModel(unittest.TestCase):
    def test_model_training(self):
        # Load the saved clustering model
        model = joblib.load('model/model.pkl')
        
        # Check model type
        self.assertIsInstance(model.named_steps['kmeans'], KMeans)
        
        # Check number of clusters >= 2
        self.assertGreaterEqual(model.named_steps['kmeans'].n_clusters, 2)
        
        # Check model has cluster centers
        self.assertEqual(
            model.named_steps['kmeans'].cluster_centers_.shape[1],
            2  # Expecting 2 features: Income & Spending
        )

if __name__ == '__main__':
    unittest.main()
