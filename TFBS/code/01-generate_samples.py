import argparse
import os
import pandas as pd
import random
from Bio import SeqIO
from collections import Counter
from logger import CustomLogger

# python 01-generate_samples.py --fasta_file path/to/your.fasta --peak_file path/to/peaks.csv --output_file path/to/output.csv --neg_type shuffle -- species SI/ATA

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate positive and negative samples for TF binding.")
    parser.add_argument("--fasta_file", type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--peak_file", type=str, required=True, help="Path to the CSV file containing peaks.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file.")
    parser.add_argument("--neg_type", type=str, choices=["dinuc_shuffle","shuffle", "random"], required=False,
                        default="shuffle", help="Type of negative sample generation.")
    parser.add_argument("--species", type=str, choices=["Si", "At"], required=True, help="Species type: Si or At")
    parser.add_argument("--dataset", type=str, choices=["Sun2022", "Malley2016"], required=True, help="Origin of Dataset")
    parser.add_argument("--sliding_window", type=int, required=False, default=200,
                        help="Size of sliding window for generating samples.")
    parser.add_argument("--fixed_length", type=int, required=False, default=None,
                        help="Fixed length for samples, overriding sliding_window if set.")
    parser.add_argument("--seed", type=int, required=False, default=646454, help="Seed for random number generation.")
    return parser.parse_args()

def count_nucleotide(df):
    # concatenate all sequences into one string
    all_sequences = df['sequence'].str.cat(sep='')

    # frequency of each nucleotide
    frequency = Counter(all_sequences)

    # total number of nucleotides
    total = sum(frequency.values())

    # convert counts to proportions (percentages) and round to 2 decimal places
    proportions = {base: round((count / total) * 100, 2) for base, count in frequency.items()}

    return dict(frequency), proportions

# Write statistics about data
def analyzie_data(logger, df):
    logger.log_message("**************************** Analyze processed data ****************************")
    # ALL DATA
    logger.log_message("ALL **************")
    logger.log_message(f"Shape: {df.shape}")
    df_len = df['sequence'].str.len()
    logger.log_message(f"Statistics:{df_len.describe().round(2)}")
    count, prop = count_nucleotide(df)
    logger.log_message(f"Nucleotide count:{count}")
    logger.log_message(f"Nucleotide proportions:{prop}")

    # POSITIVE DATA
    logger.log_message("POSITIVE **************")
    pos_df = df[df['label'] == 1]
    logger.log_message(f"Shape: {pos_df.shape}")
    pos_len = pos_df['sequence'].str.len()
    logger.log_message(f"Statistics:{pos_len.describe().round(2)}")
    count, prop = count_nucleotide(pos_df)
    logger.log_message(f"Nucleotide count:{count}")
    logger.log_message(f"Nucleotide proportions:{prop}")

    # NEGATIVE DATA
    logger.log_message("NEGATIVE **************")
    neg_df = df[df['label'] == 0]
    logger.log_message(f"Shape: {neg_df.shape}")
    neg_len = neg_df['sequence'].str.len()
    logger.log_message(f"Statistics:{neg_len.describe().round(2)}")
    count, prop = count_nucleotide(neg_df)
    logger.log_message(f"Nucleotide count:{count}")
    logger.log_message(f"Nucleotide proportions:{prop}")


# Substitute non-standard nucleotide
def convert_non_standard_to_N(sequence):
    substitutions = {'W': 'N', 'S': 'N', 'K': 'N', 'Y': 'N', 'R': 'N', 'M': 'N', 'B': 'N', 'D': 'N', 'H': 'N', 'V': 'N'}
    result = ''.join([substitutions.get(base, base) for base in sequence])

    return result

# Check all N
def all_N(sequence):
    return all(base == 'N' for base in sequence)

# Process chromosome ID based on species
def process_chrom_id(chrom, species):
    if species == "Si":
        return f"lcl|{chrom}"
    elif species == "At":
        return chrom.replace("Chr", "").replace("chr", "")
    return chrom

def load_fasta_sequences(fasta_file):
    return SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

# Calculate required padding to reach fixed_length if fixed_length is not 0
def calculate_padding(ind, peak_region_length, sliding_window, fixed_length):
    if fixed_length is not None:
        if peak_region_length > fixed_length:
            raise ValueError(
                f"Index {ind + 1}: Peak region length ({peak_region_length}) is larger than fixed_length ({fixed_length})")
        total_padding = fixed_length - peak_region_length
        left_pad = total_padding // 2
        right_pad = total_padding - left_pad
    else:  # Use sliding_window
        left_pad = sliding_window // 2
        right_pad = sliding_window - left_pad

    return left_pad, right_pad

# Extract sequence
def extract_sequence(peak_length, fixed_length, chromosome_sequence, start, end, left_pad, right_pad):
    sequence_start = max(0, start - left_pad)
    sequence_end = min(len(chromosome_sequence), end + right_pad)
    sequence_len = sequence_end - sequence_start + 1

    # try to get more from the other side when fixed_length and couldn't get enough padding on one side
    if fixed_length is not None and sequence_len < fixed_length:
        missing = fixed_length - sequence_len
        if sequence_start == 0:  # Couldn't get more on left, try right
            sequence_end = min(len(chromosome_sequence), end + right_pad + missing)
        else:  # couldn't get more on right, try left
            sequence_start = max(0, start - left_pad - missing)

    sequence = chromosome_sequence[sequence_start:sequence_end+1]
    peak_start_ind_in_padded_seq = start - sequence_start
    peak_end_ind_in_padded_seq = peak_start_ind_in_padded_seq + peak_length - 1 # end_ind is inclusive
    return sequence, peak_start_ind_in_padded_seq, peak_end_ind_in_padded_seq

# Generate positive samples
def generate_positive_samples(logger, peaks_df, fasta_sequences, species, sliding_window, fixed_length):
    logger.log_message("Generating positive samples...")
    positive_samples, peaks, start_end_inds = [], [], []
    not_found = 0

    for ind, row in peaks_df.iterrows():
        chrom_id = process_chrom_id(row["ChrID"], species)
        start, end = row["start"], row["end"]
        peak_length = end - start + 1 # end_ind is inclusive

        if chrom_id not in fasta_sequences:
            logger.log_message(f"Index {ind + 1}: {chrom_id} not in fasta file!")
            not_found += 1
            continue

        left_pad, right_pad = calculate_padding(ind, peak_length, sliding_window, fixed_length)

        # Extract sequence with appropriate padding
        chromosome_sequence = fasta_sequences[chrom_id].seq
        sequence, peak_start_ind_in_padded_seq, peak_end_ind_in_padded_seq\
            = extract_sequence(peak_length, fixed_length, chromosome_sequence, start, end, left_pad, right_pad)

        sequence = convert_non_standard_to_N(sequence)
        if all_N(sequence):
            logger.log_message(f"Row {ind + 1}: All bases are N, ignoring the row...")
        else:
            positive_samples.append((sequence.upper(), chrom_id))
            peaks.append(chromosome_sequence[start:end+1])  # end_ind is inclusive
            start_end_inds.append((peak_start_ind_in_padded_seq, peak_end_ind_in_padded_seq))


    logger.log_message(f"{not_found} out of {len(peaks_df)} rows not found in fasta file.")
    logger.log_message(f"Generated {len(positive_samples)} positive samples.")
    return positive_samples, peaks, start_end_inds

def create_negative_sequence(neg_type, sequence, all_sequences):
    if neg_type == "shuffle":
        neg_seq = "".join(random.sample(sequence, len(sequence)))  # Shuffle sequence: picks all letters but in random order
    elif neg_type == "dinuc_shuffle":
        from ushuffle import shuffle
        neg_seq = shuffle(sequence.encode('utf-8'), 2).decode('utf-8')
    elif neg_type == "random":
        # pick a random start position within all_sequences
        start = random.randint(0, len(all_sequences) - len(sequence))
        neg_seq = all_sequences[start:start + len(sequence)]
    else:
        raise ValueError(f"Given neg_type:{neg_type} is invalid!")
    return neg_seq

def generate_negative_samples(logger, positive_samples, neg_type, fasta_sequences=None):
    logger.log_message("Generating negative samples...")
    negative_samples = []

    # concatenate all chromosome sequences in fasta_sequences
    all_sequences = "".join(str(record.seq) for record in fasta_sequences.values())

    for sequence, chrom_id in positive_samples:
        neg_seq = create_negative_sequence(neg_type, sequence, all_sequences)
        negative_samples.append((neg_seq, chrom_id))

    logger.log_message(f"Generated {len(negative_samples)} negative samples.")
    return negative_samples

if __name__ == "__main__":
    args = parse_arguments()

    # Set random seed for reproducibility
    random.seed(args.seed)
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    log_name = os.path.splitext(args.output_file.split('/')[-1])[0]
    logger = CustomLogger(__name__, log_directory=log_dir, log_file=f'{log_name}.log')
    logger.log_message(f"Args: {args}")

    # load input files
    fasta_sequences = load_fasta_sequences(args.fasta_file)
    logger.log_message(f"Loaded FASTA file {args.fasta_file} with {len(fasta_sequences)} chromosome sequences.")

    peaks_df = pd.read_csv(args.peak_file)
    logger.log_message(f"Loaded peak file {args.peak_file} with {len(peaks_df)} sequences.")

    positive_samples, peaks, peak_start_end_inds_in_padded_seq = generate_positive_samples(logger, peaks_df, fasta_sequences, args.species,
                                                        args.sliding_window, args.fixed_length)
    negative_samples = generate_negative_samples(logger, positive_samples, args.neg_type, fasta_sequences)

    # combine positive and negative samples into a DataFrame
    data = pd.DataFrame({
        "species": [args.species] * (len(positive_samples) + len(negative_samples)),
        "chromosomeId": [chrom for _, chrom in positive_samples + negative_samples],
        'dataset': [args.dataset] * (len(positive_samples) + len(negative_samples)),
        "peak": [peak for peak in peaks + peaks],
        "peak_start_end_index": [start_end_ind for start_end_ind in
                                 peak_start_end_inds_in_padded_seq + peak_start_end_inds_in_padded_seq],
        "sequence": [seq for seq, _ in positive_samples + negative_samples],
        "label": [1] * len(positive_samples) + [0] * len(negative_samples)
    })
    # add unique identifier
    data.insert(0, 'uid', range(1, len(data) + 1))

    data.to_csv(args.output_file, index=False)
    logger.log_message(f"Output saved to {args.output_file} with {len(data)} samples.")

    analyzie_data(logger, data)