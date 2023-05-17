import numpy as np
import pandas as pd

from src.logger import logging  # Importing custom logger module
from src.exception import CustomException  # Importing custom exception module

import os
import sys
from src.utils import load_object


class PredictPipeline:
    """
    A pipeline for making predictions using pre-trained models.

    Attributes:
        None
    """

    def __init__(self):
        """
        Initializes a PredictPipeline object.
        """
        pass

    def predict(self, features):
        """
        Predicts the target variable using the provided features.

        Args:
            features: A list or array-like object containing the input features.

        Returns:
            The predicted target variable.

        Raises:
            CustomException: If an exception occurs during the prediction process.
        """
        try:
            # Paths to pre-trained models
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            # Load preprocessor and model
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Preprocess features
            data_scaled = preprocessor.transform(features)

            # Make predictions
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)


class CustomData:
    """
    Represents custom data for prediction.

    Attributes:
        carat: The carat value.
        depth: The depth value.
        table: The table value.
        x: The x value.
        y: The y value.
        z: The z value.
        cut: The cut value.
        color: The color value.
        clarity: The clarity value.
    """

    def __init__(self,
                 carat: float,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float,
                 cut: str,
                 color: str,
                 clarity: str):
        """
        Initializes a CustomData object with the specified attributes.

        Args:
            carat: The carat value.
            depth: The depth value.
            table: The table value.
            x: The x value.
            y: The y value.
            z: The z value.
            cut: The cut value.
            color: The color value.
            clarity: The clarity value.
        """
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        """
        Converts the custom data to a pandas DataFrame.

        Returns:
            A pandas DataFrame containing the custom data.

        Raises:
            CustomException: If an exception occurs during the DataFrame creation process.
        """
        try:
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Created")
            return df
        except Exception as e:
            logging.info("Exception Occurred in Prediction Pipeline")
            raise CustomException(e, sys)
