from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src.utils import save_object, evaluate_model

from src.logger import logging
from src.exception import CustomException

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    """
    Configuration class for the ModelTrainer.

    Attributes:
        train_model_file_path (str): The file path to save the trained model.
    """
    train_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """
    Configuration class for the ModelTrainer.

    Attributes:
        train_model_file_path (str): The file path to save the trained model.
    """

    def __init__(self):
        self.model_training_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        """
        Train and evaluate multiple regression models to find the best performing model.

        Args:
            train_array (numpy.ndarray): Training data array with features and target variable.
            test_array (numpy.ndarray): Testing data array with features and target variable.

        Raises:
            CustomException: If an exception occurs during model training and evaluation.
        """
        try:
            logging.info(
                "Splitting Dependent and Independent variables from train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define the regression models to evaluate
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elasticnet': ElasticNet(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTree': DecisionTreeRegressor(),
                'RandomForest': RandomForestRegressor()
            }

            # Evaluate the performance of each model
            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models)
            logging.info(f"Model Report: {model_report}")

            # Find the best performing model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(
                model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logging.info(
                f"Best Model Found, Model Name: {best_model_name}, R2_Score: {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_training_config.train_model_file_path,
                obj=best_model
            )
        except Exception as e:
            raise CustomException(e, sys)
