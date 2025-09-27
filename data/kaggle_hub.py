import logging
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter
from pandas import DataFrame

logger = logging.getLogger(__name__)

class KaggleHub:
  def __init__(self):
    self.adapter = KaggleDatasetAdapter.PANDAS

  def load_dateset(self, handle, file_path) -> DataFrame:
    logger.info(f"Loading dataset {file_path}")
    try:
      df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        handle,
        file_path
      )
      logger.info(f"Load dataset successfully, {file_path}")
      logger.info(f"Data info: {df.shape}")
      return df
    except Exception as e:
      logger.error(f"Failed to load dataset, {file_path}", e)
      raise e

  def save_dataset_to_csv(self, df: DataFrame, file_name: str) -> None:
    logger.info(f"Saving dataset as {file_name}")
    try:
      file = Path(file_name)
      if file.exists():
        logger.info(f"File {file_name} already exists. Save operation aborted")
      else:
        df.to_csv(file_name, index=False)
        logger.info(f"Save dataset successfully, {file_name}")
    except Exception as e:
      logger.error(f"Failed to save dataset, {file_name}", e)
      raise e