import argparse
import pandas as pd
import random
from Bio import SeqIO


"""
Example usage

python 01-generate_samples.py --fasta_file path/to/your.fasta --peak_file path/to/peaks.csv --output_file path/to/output.csv --neg_type shuffle -- species SI/ATA

python 01-generate_samples.py --fasta_file ../inputs/fastas/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa --peak_file ../inputs/peak_files/AtABFs_DAP-Seq_peaks.csv --species "At" --dataset Josey  --output_file ../inputs/AtABFs_training_shuffle_neg_stride_200.csv’
python 01-generate_samples.py --fasta_file ../inputs/fastas/Si_sequence --peak_file ../inputs/peak_files/SiABFs_DAP-Seq_peaks.csv --species "Si" --dataset Josey  --output_file ../inputs/SiABFs_training_shuffle_neg_stride_200.csv  
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate positive and negative samples for TF binding.")
    parser.add_argument("--fasta_file", type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--peak_file", type=str, required=True, help="Path to the CSV file containing peaks.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file.")
    parser.add_argument("--neg_type", type=str, choices=["dinuc_shuffle","shuffle", "random", "matched"], required=False, default="shuffle")
    parser.add_argument("--bed_path", type=str, required=False, default=".")
    parser.add_argument("--species", type=str, choices=["Si", "At"], required=True, help="Species type: Si or At")
    parser.add_argument("--dataset", type=str, choices=["Josey", "Ronan"], required=True, help="Origin of Dataset")
    parser.add_argument("--sliding_window", type=int, required=False, default=200)
    parser.add_argument("--fixed_length", type=int, required=False, default=None,
                        help="generates samples of fixed_length, ignoring sliding_window;"
                             "if set to None, would use sliding_window and length would vary based on original length of peak")
    return parser.parse_args()

def convert_to_N(sequence):
    substitutions = {'W': 'N', 'S': 'N', 'K': 'N', 'Y': 'N', 'R': 'N', 'M': 'N', 'B': 'N', 'D': 'N', 'H': 'N', 'V': 'N'}
    result = ''.join([substitutions.get(base, base) for base in sequence])

    return result

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
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    return fasta_sequences


def generate_positive_samples(peaks_df, fasta_sequences, species, sliding_window, fixed_length):
    positive_samples = []
    peaks = []
    not_found = 0

    for ind, row in peaks_df.iterrows():
        # get sequence coordinates
        chrom_id = process_chrom_id(row["ChrID"], species)

        start = row["start"]
        end = row["end"]
        peak_region_length = end - start + 1

        # calculate required padding to reach fixed_length if fixed_length is not 0
        if fixed_length is not None:
            if peak_region_length > fixed_length:
                raise ValueError(
                    f'index {ind + 1}: peak_region_length ({peak_region_length}) is larger than fixed_length ({fixed_length})')

            total_padding = fixed_length - peak_region_length
            left_pad = total_padding // 2
            right_pad = total_padding - left_pad
        else: # use sliding_window
            left_pad = sliding_window // 2
            right_pad = sliding_window - left_pad
        
        if chrom_id in fasta_sequences:
            full_sequence = fasta_sequences[str(chrom_id)].seq

            # Extract sequence with appropriate padding
            sequence_start = max(0, start - left_pad)
            sequence_end = min(len(full_sequence), end + right_pad)
            sequence = full_sequence[sequence_start:sequence_end]

            # if fixed_length and couldn't get enough padding on one side, try to get more from the other side
            if fixed_length is not None and len(sequence) < fixed_length:
                missing = fixed_length - len(sequence)
                if sequence_start == 0:  # Couldn't get more on left, try right
                    sequence_end = min(len(full_sequence), end + right_pad + missing)
                else:  # Couldn't get more on right, try left
                    sequence_start = max(0, start - left_pad - missing)
                sequence = full_sequence[sequence_start:sequence_end]
            
            # change all non-valid nucleotides to N
            sequence = convert_to_N(sequence)

            if all_N(sequence):
                print(f'all bases for row {ind+1} in peak file is N.\nIgnoring the row...')
            else:
                positive_samples.append(((str(sequence)).upper(), chrom_id))  # Include chrom_id
                peaks.append(full_sequence[start:end])
        else:
            print(f"{chrom_id} not in fasta file!")
            not_found +=1

    if not_found == 0:
        print("all rows in peak file has been found in fasta file...")
    else:
        print(f"sequence regarding to {not_found} / {len(peaks_df)} rows has not been found in fasta file...")

    return positive_samples, peaks


def generate_negative_samples(positive_samples, neg_type, fasta_sequences=None):
    negative_samples = []
    if neg_type == "random":
        # concatenate all sequences in fasta_sequences
        all_sequences = "".join(
            str(record.seq) for record in fasta_sequences.values())

    for sequence, chrom_id in positive_samples:
        # shuffling the original sequence
        if neg_type == "shuffle":
            neg_seq = "".join(random.sample(sequence, len(sequence))) # picks all letters but in random order
        elif neg_type == "dinuc_shuffle":
            from ushuffle import shuffle, Shuffler
            neg_seq = shuffle(sequence.encode('utf-8'), 2).decode('utf-8')
        # picking a random sequence from the fasta_sequences
        elif neg_type == "random":
            # pick a random start position within all_sequences
            start = random.randint(0, len(all_sequences) - len(sequence))
            neg_seq = all_sequences[start:start + len(sequence)]
        else:
            raise  ValueError(f"Given neg_type:{neg_type} is invalid!")

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
    if args.neg_type == "matched":
        bed_path = args.bed_path
        with open(bed_path, 'w') as bed_file:
            for _, row in peaks_df.iterrows():
                chrom = row['ChrID']
                start = int(row['start']) - 1  # convert to 0-based
                end = int(row['end'])
                bed_file.write(f"{chrom}\t{start}\t{end}\n")
    else:
        positive_samples, peaks = generate_positive_samples(peaks_df, fasta_sequences, args.species,
                                                 args.sliding_window, args.fixed_length)
        print(f"number of positive samples: {len(positive_samples)}")
        print("Generating negative samples...")
        negative_samples = generate_negative_samples(positive_samples, args.neg_type, fasta_sequences)
        print(f"number of negative samples: {len(negative_samples)}")

        # Combine into a DataFrame and save
        data = pd.DataFrame({
            "species": [args.species] * (len(positive_samples) + len(negative_samples)),
            "chromosomeId": [chrom for _, chrom in positive_samples + negative_samples],
            'dataset': [args.dataset] * (len(positive_samples) + len(negative_samples)),
            "peak": [peak for peak in peaks + peaks],
            "sequence": [seq for seq, _ in positive_samples + negative_samples],
            "label": [1] * len(positive_samples) + [0] * len(negative_samples)
        })

        data.to_csv(args.output_file, index=False)
        print(f"Output saved to {args.output_file}")


if __name__ == "__main__":
    main()
