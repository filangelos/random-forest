import unittest


class TestData(unittest.TestCase):
    """Test `data.py`"""

    def test__getData_Toy_Spiral(self):
        """Test `getData('Toy_Spiral')` function."""
        from src.data import getData
        import matplotlib.pyplot as plt
        # fetch data
        data_train, data_query = getData('Toy_Spiral')
        # training data
        plt.scatter(data_train[:, 0], data_train[:, 1], c=data_train[:, 2])
        # testing data - meshgrid
        plt.scatter(data_query[:, 0], data_query[:, 1],
                    c=data_query[:, 2], alpha=0.05)
        # show plot for 2 seconds
        plt.draw()
        plt.pause(2)
        return self.assertEqual(data_train.shape[1], data_query.shape[1])


if __name__ == '__main__':
    unittest.main()
