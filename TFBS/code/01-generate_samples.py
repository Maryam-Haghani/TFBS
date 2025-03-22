import argparse
import pandas as pd
import random
from Bio import SeqIO


"""
Example usage

python 01-generate_samples.py --fasta_file path/to/your.fasta --peak_file path/to/peaks.csv --output_file path/to/output.csv --neg_type shuffle -- species SI/ATA

python 01-generate_samples.py --fasta_file ../inputs/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa --peak_file ../inputs/ata_peaks.csv --output_file ../inputs/ata_training_shuffle_neg_stride_200.csv

"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate positive and negative samples for TF binding.")
    parser.add_argument("--fasta_file", type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--peak_file", type=str, required=True, help="Path to the CSV file containing peaks.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file.")
    parser.add_argument("--neg_type", type=str, choices=["shuffle", "random"], required=False, default="shuffle")
    parser.add_argument("--species", type=str, choices=["SI", "ATA"], required=False,  default="ATA", help="Species type: SI or ATA")
    parser.add_argument("--sliding_window", type=int, required=False, default=200)
    return parser.parse_args()


def convert_to_N(sequence):
    substitutions = {'W': 'N', 'S': 'N', 'K': 'N', 'Y': 'N', 'R': 'N', 'M': 'N', 'B': 'N', 'D': 'N', 'H': 'N', 'V': 'N'}
    result = ''.join([substitutions.get(base, base) for base in sequence])

    return result

def all_N(sequence):
    return all(base == 'N' for base in sequence)

# Process chromosome ID based on species
def process_chrom_id(chrom, species):
    if species == "SI":
        return f"lcl|{chrom}"
    elif species == "ATA":
        return chrom.replace("Chr", "")
    return chrom

def load_fasta_sequences(fasta_file):
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    return fasta_sequences


def generate_positive_samples(peaks_df, fasta_sequences, species, sliding_window):
    positive_samples = []
    for ind, row in peaks_df.iterrows():
        # get sequence coordinates
        chrom_id = process_chrom_id(row["ChrID"], species)
        start = row["start"]
        end = row["end"]
        
        if chrom_id in fasta_sequences:
            full_sequence = fasta_sequences[str(chrom_id)].seq
            
            # change all non-valid nuclotides to N
            sequence = full_sequence[max(0, start-sliding_window//2):min(len(full_sequence), end+sliding_window//2)]
            sequence = convert_to_N(sequence)

            if all_N(sequence):
                print(f'all bases for row {ind+1} in peak file is N.\nIgnoring the row...')
            else:
                positive_samples.append(((str(sequence)).upper(), chrom_id))  # Include chrom_id

    return positive_samples


def generate_negative_samples(positive_samples, neg_type, fasta_sequences=None):
    negative_samples = []

    for sequence, chrom_id in positive_samples:
        if neg_type == "shuffle":
            neg_seq = "".join(random.sample(sequence, len(sequence)))
        elif neg_type == "random":
            all_sequences = "".join(str(record.seq) for record in fasta_sequences.values())
            start = random.randint(0, len(all_sequences) - len(sequence))
            neg_seq = all_sequences[start:start + len(sequence)]

        negative_samples.append((neg_seq, chrom_id))

    return negative_samples


def main():
    args = parse_arguments()

    # Load input files
    fasta_sequences = load_fasta_sequences(args.fasta_file)
    print(f"fasta sequence file has been loaded with {len(fasta_sequences)} sequences")
    peaks_df = pd.read_csv(args.peak_file)
    print(f"peak file has been loaded with {len(peaks_df)} sequences")

    # Generate positive and negative samples
    print("Generating positive samples...")
    positive_samples = generate_positive_samples(peaks_df, fasta_sequences, args.species, args.sliding_window)
    print(f"number of positive samples: {len(positive_samples)}")

    print("Generating negative samples...")
    negative_samples = generate_negative_samples(positive_samples, args.neg_type, fasta_sequences)
    print(f"number of negative samples: {len(negative_samples)}")

    # Combine into a DataFrame and save
    data = pd.DataFrame({
        "sequence": positive_samples + negative_samples,
        "label": [1] * len(positive_samples) + [0] * len(negative_samples)
    })

    data = pd.DataFrame({
        "chromosomeId": [chrom for _, chrom in positive_samples + negative_samples],
        "sequence": [seq for seq, _ in positive_samples + negative_samples],
        "label": [1] * len(positive_samples) + [0] * len(negative_samples)
    })
    
    data.to_csv(args.output_file, index=False)
    print(f"Output saved to {args.output_file}")


if __name__ == "__main__":
    main()
