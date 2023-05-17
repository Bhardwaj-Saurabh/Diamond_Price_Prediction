import os
import sys
from src.logger import logging  # Importing custom logger module
from src.exception import CustomException  # Importing custom exception module

import pandas as pd  # Importing pandas library 
from sklearn.model_selection import train_test_split  

# Importing dataclass decorator from dataclasses module
from dataclasses import dataclass  

## Data Ingestion configuration

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    """
    raw_data_path: str =  os.path.join('artifacts', 'raw.csv')  # Default path for raw data
    train_data_path: str = os.path.join('artifacts', 'train.csv')  # Default path for train data
    test_data_path: str = os.path.join('artifacts', 'train.csv')  # Default path for test data

class DataIngestion:
    """
    Class for performing data ingestion.
    """
    def __init__(self) -> None:
        """
        Initializes the DataIngestion object with the default configuration.
        """
        self.ingestion_config = DataIngestionConfig()  # Initializing the ingestion configuration
    
    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process.
        
        Reads the data from a CSV file, splits it into train and test sets,
        and saves the data as CSV files. Logs the progress and handles any exceptions.
        
        Returns:
            Tuple[str, str]: Paths of the train and test data CSV files.
        
        Raises:
            CustomException: If an exception occurs during the data ingestion process.
        """
        logging.info("Data Ingestion started")  # Logging info message
        
        try:
            # Read the data from the CSV file
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            logging.info('Data read as Pandas DataFrame')  # Logging info message

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            # Save the raw data as CSV file
            df.to_csv(self.ingestion_config.raw_data_path)

            logging.info('Raw data stored as CSV file')  # Logging info message

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            # Save the train data as CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Save the test data as CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed")  # Logging info message

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occurred at Data Ingestion Stage")  # Logging info message
            raise CustomException(e, sys)  
