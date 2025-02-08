import unittest
import kaboudan
import pandas as pd
import numpy as np

class TestKaboudan(unittest.TestCase):

    def test_kaboudan_metric(self):
        sse_original = 100
        sse_shuffled = 200
        result = kaboudan.kaboudan_metric(sse_original, sse_shuffled)
        self.assertTrue(0 <= result <= 1)

    def test_analyze_series(self):
        # Create a dummy series with a DatetimeIndex for testing
        dates = pd.date_range(start='2020-01-01', periods=100, freq='M')
        data = pd.Series(np.random.rand(100), index=dates)
        result = kaboudan.analyze_series('Test', data)
        
        self.assertIsInstance(result, dict)
        self.assertTrue('name' in result)
        self.assertTrue('sse_original' in result)
        self.assertTrue('sse_shuffled' in result)
        self.assertTrue('kaboudan' in result)
        self.assertTrue('series_returns' in result)
        self.assertTrue('train_returns' in result)
        self.assertTrue('train_shuffled_returns' in result)
        self.assertTrue('test_returns' in result)
        self.assertTrue('y_pred_returns' in result)
        self.assertTrue('y_pred_shuffled_returns' in result)

if __name__ == '__main__':
    unittest.main()
