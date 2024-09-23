import unittest
from typing import List
from mlcv.datasets.fetcher import transform_data, load_numeric_data, load


class TestTransformData(unittest.TestCase):

    def test_list_length(self):
        result = load_numeric_data()
        self.assertEqual(len(result), 100, "List should contain 100 elements")

    def test_load_numeric_with_transformation(self):
        """Test loading numeric data with transformation."""
        result = load(dataset_name='numeric', allow=True, operation=['square'], chunk_size=3, operations_count=1)
        self.assertIsInstance(result, list, "Result should be a list")
        self.assertEqual(len(result), 100, "Result should have 100 elements")
        self.assertNotEqual(result, list(range(1, 101)), "Result should be transformed and not equal to original data")

    def test_square_operation(self):
        data = [1, 2, 3, 4, 5]
        result = transform_data(data, ['square'])
        expected = [1, 8, 9, 64, 25]
        self.assertEqual(result, expected)

    def test_fibonacci_operation(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = transform_data(data, ['fibonacci'])
        # Prime indices are 2, 3, 5, 7
        expected = [1,  2,  1, 2, 5,  5,  7, 13, 9,10  ]
        self.assertEqual(result, expected)

    def test_accumulate(self):
        # 0 1 1 2 3 5 8 13
        data = [1, 2, 3, 4]
        result = transform_data(data, ['accumulate'])
        # Prime indices are 2, 3, 5, 7
        expected = [1,3,6,10]
        self.assertEqual(result, expected)

    def test_reverse_chunks_operation(self):
        data = [1, 2, 3, 4, ]
        result = transform_data(data, ['reverse_chunks'], chunk_size=4)
        expected = [4,3,2,1]
        self.assertEqual(result, expected)

    def test_recursive_sum(self):
        data = [100]
        result = transform_data(data, ['recursive_sum_digits'], chunk_size=4)
        expected = [1]
        self.assertEqual(result, expected)

    def test_rotate_by_prime(self):
        data = [1,2,3,4]
        result = transform_data(data, ['rotate_by_prime'], chunk_size=4)
        expected = [3,4,1,2]
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()