import unittest


class TestData(unittest.TestCase):
    """Test `data.py`"""

    def test__plot_toydata(self):
        """Test `plot_toydata` function."""
        from src.data import getData
        from src.visualise import plot_toydata
        import matplotlib.pyplot as plt
        # fetch data
        data_train, data_query = getData('Toy_Spiral')
        # training data
        plot_toydata(data_train)
        # test data
        plot_toydata(data_query, alpha=0.05)
        # show plot for 2 seconds
        plt.draw()
        plt.pause(2)
        return self.assertIsNone(None)


if __name__ == '__main__':
    unittest.main()
