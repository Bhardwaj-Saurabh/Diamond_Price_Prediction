import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def save_object(file_path, obj):
    """
    Save an object to a file using pickle.

    Args:
        file_path (str): Path to the file where the object will be saved.
        obj: The object to be saved.

    Raises:
        CustomException: If an exception occurs during object saving.
    """
    try:
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the file
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the performance of machine learning models.

    Args:
        X_train (numpy.ndarray): Training input features.
        y_train (numpy.ndarray): Training target variable.
        X_test (numpy.ndarray): Testing input features.
        y_test (numpy.ndarray): Testing target variable.
        models (dict): Dictionary containing the models to evaluate.

    Returns:
        dict: A dictionary containing the evaluation report with model names as keys and evaluation scores as values.

    Raises:
        CustomException: If an exception occurs during model evaluation.
    """
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # Fit the model on the training data
            model.fit(X_train, y_train)

            # Make predictions on the testing data
            y_test_pred = model.predict(X_test)

            # Calculate the evaluation scores
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load an object from a file using pickle.

    Args:
        file_path (str): The file path of the object to load.

    Returns:
        The loaded object.

    Raises:
        CustomException: If an error occurs while loading the object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
