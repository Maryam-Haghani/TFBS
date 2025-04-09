import pandas as pd
from collections import Counter
import argparse
from logger import CustomLogger

# python analyze_data.py --generated_data_file '../inputs/AtABFs_training_shuffle_neg_stride_200.csv' --species "At"
# python analyze_data.py --generated_data_file '../inputs/SiABFs_training_shuffle_neg_stride_200.csv' --species "Si"
# python analyze_data.py --generated_data_file '../inputs/Josey-AtABF2_training_shuffle_neg_stride_201.csv' --dataset "ABF2-Josey-201"
# python analyze_data.py --generated_data_file '../inputs/Ronan-AtABF2_training_shuffle_neg_stride_201.csv' --dataset "ABF2-Ronan-201"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze generated dataset")
    parser.add_argument("--generated_data_file", type=str, required=True, help="Path to the CSV file containing generated samples")
    parser.add_argument("--species", type=str, choices=["Si", "At"], required=False,  default="At", help="Species type: Si or At")
    parser.add_argument("--dataset", type=str, required=True)
    return parser.parse_args()

def count_nucleotide(df):
    # Concatenate all sequences into one string
    all_sequences = df['sequence'].str.cat(sep='')

    # Count the frequency of each nucleotide
    frequency = Counter(all_sequences)

    # Calculate total number of nucleotides
    total = sum(frequency.values())

    # Convert counts to proportions (percentages) and round to 2 decimal places
    proportions = {base: round((count / total) * 100, 2) for base, count in frequency.items()}

    return dict(frequency), proportions


args = parse_arguments()

df = pd.read_csv(args.generated_data_file)

logger = CustomLogger(__name__, log_directory=f'../outputs/logs', log_file = f'log_data_{args.species}-{args.dataset}.log')

# ALL DATA
logger.log_message("************** ALL **************")
logger.log_message(f"shape: {df.shape}")
df_len = df['sequence'].str.len()
logger.log_message(f"statistics:{df_len.describe().round(2)}")
count, prop = count_nucleotide(df)
logger.log_message(f"nucleotide count:{count}")
logger.log_message(f"nucleotide proportions:{prop}")


# POSITIVE DATA
logger.log_message("************** POSITIVE **************")
pos_df = df[df['label'] == 1]
logger.log_message(f"shape: {pos_df.shape}")
pos_len = pos_df['sequence'].str.len()
logger.log_message(f"statistics:{pos_len.describe().round(2)}")
count, prop = count_nucleotide(pos_df)
logger.log_message(f"nucleotide count:{count}")
logger.log_message(f"nucleotide proportions:{prop}")

# NEGATIVE DATA
logger.log_message("************** NEGATIVE **************")
neg_df = df[df['label'] == 0]
logger.log_message(f"shape: {neg_df.shape}")
neg_len = neg_df['sequence'].str.len()
logger.log_message(f"statistics:{neg_len.describe().round(2)}")
count, prop = count_nucleotide(neg_df)
logger.log_message(f"nucleotide count:{count}")
logger.log_message(f"nucleotide proportions:{prop}")