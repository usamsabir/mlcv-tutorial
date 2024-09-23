import numpy as np
import os
from sklearn import datasets
from typing import List

DATA_DIR = os.path.split(os.path.realpath(__file__))[0]
DATASETS = ['iris', 'banknote', 'old_faithful', 'greetings', 'numeric']


def load_iris():
    """Load a data set for multi-class classification.

    Returns
    -------
    X : array (150, 4)
        The data samples.

    y : array (150,)
        The class labels.
    """
    return datasets.load_iris(return_X_y=True)


def load_banknote():
    """Load a data set for binary classification.

    Returns
    -------
    X : array (1348, 4)
        The data samples.

    y : array (1348,)
        The (binary) class labels.
    """

    data_path = os.path.join(DATA_DIR, 'banknote_auth_data.csv')
    label_path = os.path.join(DATA_DIR, 'banknote_auth_labels.csv')

    X = np.loadtxt(data_path, delimiter=',')
    y = np.loadtxt(label_path, dtype=str)

    return X, y


def load_old_faithful():
    """Load a data set for clustering.

    Returns
    -------
    X : array (272, 2)

    y : None

    """
    data_path = os.path.join(DATA_DIR, 'old_faithful.txt')
    X = np.loadtxt(data_path, skiprows=1, usecols=(1, 2))

    return X, None


def load_greetings():
    """Load a welcoming data set.

    Returns
    -------
    X : array (11324, 3)

    """

    data_path = os.path.join(DATA_DIR, 'greetings.txt')
    X = np.loadtxt(data_path, delimiter=',')

    return X, None


def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

def recursive_sum_digits(num):
    while num > 9:
        num = sum(int(digit) for digit in str(num))
    return num
def transform_data(data: List[int], operation: List[str], chunk_size: int = 3, operations_count: int = 1) -> List[int]:
    """
    Transforms a list of integers based on specified operations.

    Args:
        data (List[int]): The input list of integers.
        operation (List[str]): A list of operations to perform. Supported operations are:
            - 'square': Square values at even indices and cube values at odd indices.
            - 'fibonacci': Replace values at prime number indices with their corresponding Fibonacci number.
            - 'reverse_chunks': Reverse every chunk of 'chunk_size' elements in the list.
            - 'accumulate': Replace each element with the sum of all previous elements and itself.
            - 'rotate_by_prime': Rotate the entire list to the right by the number of positions equal to the count of prime numbers in the list.
            - 'recursive_sum_digits': Repeatedly sum the digits of each number until a single digit is reached.
        chunk_size (int, optional): The size of chunks for 'reverse_chunks' operation. Defaults to 3.
        operations_count (int, optional): Number of times to repeat each operation. Defaults to 1.

    Returns:
        List[int]: The transformed list of integers.
    """
    for _ in range(operations_count):
        for op in operation:
            if op == 'square':
                data = [data[i] ** 2 if i % 2 == 0 else data[i] ** 3 for i in range(len(data))]
            elif op == 'fibonacci':
                data = [fibonacci(i) if is_prime(i) else data[i] for i in range(len(data))]
            elif op == 'reverse_chunks':
                data = [data[i:i + chunk_size][::-1] + data[i + chunk_size:] for i in range(0, len(data), chunk_size)][
                    0]
            elif op == 'accumulate':
                for i in range(1, len(data)):
                    data[i] += data[i - 1]
            elif op == 'rotate_by_prime':
                prime_count = sum([1 for num in data if is_prime(num)])
                data = data[-prime_count:] + data[:-prime_count]
            elif op == 'recursive_sum_digits':
                data = [recursive_sum_digits(num) for num in data]
    return data


def load_numeric_data() -> List[int]:
    """
        Generate a list of numbers from 1 to 100.

        Returns:
            List[int]: A list containing numbers from 1 to 100.
    """
    return list(range(1, 101))


def load(dataset_name: str = 'numeric', allow: bool = False, operation: List[str] = ['square'], chunk_size: int = 3,
         operations_count: int = 1):
    """Load a data set.

    Parameters
    ----------
    dataset_name : str
        Name of the data set to load.
    allow: bool
        if True then data will be loaded from load_numeric_data() and then transformed using transform_data
    operation : List[str]
        Name of the operation to perform on the data
    chunk_size : int
        Size of the chunk
    operations_count : int
        Number of times to repeat each operation.

    Returns
    -------
    X : array, shape (n_samples, n_features)
        The data samples.

    y : array, shape (n_samples,) (optional)
        The data targets if there are any.

    """

    dataset_name = dataset_name.lower()
    if dataset_name not in DATASETS:
        raise ValueError('Dataset {} unknown.\nSupproted datasets:\n{}'
                         .format(dataset_name, DATASETS))

    if dataset_name == 'iris':
        return load_iris()
    elif dataset_name == 'banknote':
        return load_banknote()
    elif dataset_name == 'old_faithful':
        return load_old_faithful()
    elif dataset_name == 'greetings':
        return load_greetings()
    elif dataset_name == 'numeric':
        if allow:
            data = load_numeric_data()
            return transform_data(data, operation, chunk_size, operations_count)
        else:
            return load_numeric_data()