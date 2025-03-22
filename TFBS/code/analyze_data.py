import pandas as pd
import logging
from collections import Counter

def create_logger(log_file):
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.DEBUG)

  file_handler = logging.FileHandler(log_file)
  file_handler.setLevel(logging.DEBUG)

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.DEBUG)

  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  file_handler.setFormatter(formatter)
  console_handler.setFormatter(formatter)

  logger.addHandler(file_handler)
  logger.addHandler(console_handler)
  
  return logger

  logger.addHandler(file_handler)

def count_nucleotide(df):
    # Concatenate all sequences into one string
    all_sequences = df['sequence'].str.cat(sep='')

    # Count the frequency of each nucleotide
    frequency = Counter(all_sequences)
    return dict(frequency)

# df = pd.read_csv('../inputs/ata_training_pos_shuffle_neg_200.csv')
df = pd.read_csv('../inputs/ata_training_shuffle_neg_stride_200.csv')

logger = create_logger('../outputs/logs/data.log')

# ALL DATA
logger.info("************** ALL **************")
logger.info(f"shape: {df.shape}")
df_len = df['sequence'].str.len()
logger.info(f"statistics:{df_len.describe().round(2)}")
logger.info(f"nucleotide count:{count_nucleotide(df)}")


# POSITIVE DATA
logger.info("************** POSITIVE **************")
pos_df = df[df['label'] == 1]
logger.info(f"shape: {pos_df.shape}")
pos_len = pos_df['sequence'].str.len()
logger.info(f"statistics:{pos_len.describe().round(2)}")
logger.info(f"nucleotide count:{count_nucleotide(pos_df)}")

# NEGATIVE DATA
logger.info("************** NEGATIVE **************")
neg_df = df[df['label'] == 0]
logger.info(f"shape: {neg_df.shape}")
neg_len = neg_df['sequence'].str.len()
logger.info(f"statistics:{neg_len.describe().round(2)}")
logger.info(f"nucleotide count:{count_nucleotide(neg_df)}")