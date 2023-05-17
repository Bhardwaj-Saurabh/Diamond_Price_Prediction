import numpy as np
import pandas as pd

from src.logger import logging  # Importing custom logger module
from src.exception import CustomException  # Importing custom exception module

# Importing SimpleImputer for handling missing values
from sklearn.impute import SimpleImputer
# Importing StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler
# Importing OrdinalEncoder for ordinal encoding
from sklearn.preprocessing import OrdinalEncoder

# Importing Pipeline for creating data transformation pipelines
from sklearn.pipeline import Pipeline
# Importing ColumnTransformer for column-specific transformations
from sklearn.compose import ColumnTransformer
# Importing dataclass decorator from dataclasses module
from dataclasses import dataclass

import os
import sys
# Importing custom utility function for saving objects
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    """
    preprocessor_object_file_path: str = os.path.join(
        'artifacts', 'preprocessor.pkl')


class DataTransformation:
    """
    Class for performing data transformation.
    """

    def __init__(self):
        """
        Initializes the DataTransformation object with the default configuration.
        """
        self.data_transformation_config = DataTransformationConfig(
        )  # Initializing the data transformation configuration

    def get_data_transformation_obj(self):
        """
        Obtains the data transformation object.

        Returns:
            ColumnTransformer: Preprocessing object containing the data transformation pipelines.

        Raises:
            CustomException: If an error occurs during data transformation.
        """
        try:
            # Logging info message
            logging.info('Data Transformation Initiated')

            # Categorical and numerical columns
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1',
                                  'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            # Logging info message
            logging.info("Data Transformation Pipeline Initiated")

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    # Handling missing values using median
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())  # Scaling numerical features
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    # Handling missing values using most frequent
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[
                     cut_categories, color_categories, clarity_categories])),  # Ordinal encoding
                    # Scaling categorical features
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                # Applying numerical pipeline to numerical columns
                ('num_pipeline', num_pipeline, numerical_cols),
                # Applying categorical pipeline to categorical columns
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            # Logging info message
            logging.info("Data Transformation Completed")
            return preprocessor

        except Exception as e:
            # Logging error message
            logging.error('Error occurred during Data Transformation')
            # Raising custom exception with the original exception and system information
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        """
        Initiates the data transformation process.

        Reads the train and test data, applies the data transformation pipeline,
        and saves the preprocessor object. Logs the progress and handles any exceptions.

        Args:
            train_data_path (str): Path to the train data CSV file.
            test_data_path (str): Path to the test data CSV file.

        Returns:
            Tuple[np.ndarray, np.ndarray, str]: Transformed train and test data arrays, and preprocessor object file path.

        Raises:
            CustomException: If an exception occurs during the data transformation process.
        """
        try:
            # Read the train data from CSV
            train_df = pd.read_csv(train_data_path)
            # Read the test data from CSV
            test_df = pd.read_csv(test_data_path)

            # Logging info message
            logging.info('Read Train and Test Data completed')
            # Logging train data head
            logging.info(
                f"Train Dataframe Head:\n{train_df.head().to_string()}")
            # Logging test data head
            logging.info(f"Test Dataframe Head:\n{test_df.head().to_string()}")

            # Logging info message
            logging.info("Obtaining preprocessing object")
            # Obtain the data transformation object
            preprocessing_obj = self.get_data_transformation_obj()

            target_column = 'price'
            drop_columns = [target_column, 'id']

            # Extract input features from train data
            input_feature_train_df = train_df.drop(
                columns=drop_columns, axis=1)
            # Extract target feature from train data
            target_feature_train_df = train_df[target_column]

            # Extract input features from test data
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            # Extract target feature from test data
            target_feature_test_df = test_df[target_column]

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)  # Apply preprocessing on train input features
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)  # Apply preprocessing on test input features

            # Combine preprocessed train input features and target feature
            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            # Combine preprocessed test input features and target feature
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            # Logging info message
            logging.info(
                "Applying Preprocessing Object on training and testing datasets")

            save_object(
                # Save the preprocessor object
                file_path=self.data_transformation_config.preprocessor_object_file_path,
                obj=preprocessing_obj,
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_object_file_path
            )

        except Exception as e:
            # Logging error message
            logging.error('Error occurred during Data Transformation')
            # Raising custom exception with the original exception and system information
            raise CustomException(e, sys)
