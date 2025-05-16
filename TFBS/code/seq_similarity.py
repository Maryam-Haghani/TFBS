import pandas as pd
import os
import subprocess
import shutil
import sys
from Bio import SeqIO


def create_fasta(logger, df, temp_dir):
    """
    Creates a single FASTA file from the given data.

    Each row in the DataFrame must have columns:
        - 'species'
        - 'chromosome' (or 'chromosomeId' if that's your actual column name)
        - 'origin'
        - 'label'
        - 'sequence'

    The header for each FASTA entry is formatted as:
        >species_chromosome_dataset_label

    The sequence is taken from the 'sequence' column.

    Parameters
    ----------
    df : pd.DataFrame
    temp_dir : str
    """

    output_fasta = os.path.join(temp_dir, "combined_sequences.fasta")

    if not os.path.exists(output_fasta):
        logger.log_message(f"---------Creating fasta file Fasta file")
        with open(output_fasta, "w") as fasta_file:
            for index, row in df.iterrows():
                header = f">{row['entity']}"
                sequence = str(row['sequence'])

                fasta_file.write(header + "\n")
                fasta_file.write(sequence + "\n")

        logger.log_message(f"{len(df)} sequences saved at fasta file: '{output_fasta}'")

    else:
        sequences = list(SeqIO.parse(output_fasta, "fasta"))
        logger.log_message(f"{len(sequences)} sequences loaded from file: '{output_fasta}'")

    return output_fasta

def run_cd_hit(logger, fasta_file, sim, word_size, temp_dir):
    """
    Reference: https://github.com/weizhongli/cdhit/blob/master/doc/cdhit-user-guide.wiki

    *** Run:
    cd-hit-est \
   -i input.fa          # FASTA with your DNA sequences
   -o clusters90.fa     # representative sequences
   -c 0.90              # identity threshold (here 90%)
   -n 8                 # word length that matches –c
   -M 16000             # memory limit in MB (0 = use all)
   -T 8                 # number of CPU threads
   -d 0                 # keep full headers in the output


    *** Output:
    - [cd_git_file] – one representative (“seed”) for every cluster.
    - [cd_git_file].clstr – a plain‑text log that lists every sequence and the cluster it belongs to.

    *** Validate that the CD‑HIT(‑EST) parameters *identity cut‑off* (‑c) and *word length* (‑n) are compatible.

        CD‑HIT filters candidate matches with fixed‑length k‑mers (“words”).
        For reliable results, the word size **must** match the requested
        sequence‑identity threshold; otherwise CD‑HIT aborts with an error.

        Acceptable ‑c ↔‑n pairs for **nucleotide mode** (``cd‑hit‑est``)
        ---------------------------------------------------------------
            Identity range (‑c)   Allowed word length(s) (‑n)
            -------------------   --------------------------
              0.95-1.00                 10  or 11
              0.90-0.95                 8  or  9
              0.88-0.90                     7
              0.85-0.88                     6
              0.80-0.85                     5
              0.75-0.80                     4

        Values of ``‑c`` below ~0.75 are not supported because the algorithm
        becomes insufficiently sensitive.

        Parameters
        ----------
        c_identity : float
            Desired global sequence‑identity cut‑off (0 < c≤ 1).
        n_word : int
            Length of the k‑mer word to use.

        Raises
        ------
        ValueError
            If the supplied ``c_identity``/``n_word`` combination is invalid.

        Notes
        -----
        Reference table adapted from the official CD‑HIT documentation
        (home.cc.umanitoba.ca).  Modify as needed if CD‑HIT updates its
        supported ranges in future releases.
        """

    cd_hit_file = os.path.join(temp_dir, f"clusters_c-{sim}_n-{word_size}.fa")

    if not os.path.exists(cd_hit_file):
        logger.log_message(f"Running CD-HIT-est...")

        if shutil.which("cd-hit-est") is None:
            sys.exit("cd-hit-est executable not found. Install CD‑HIT or add it to PATH.")

        cmd = [
            "cd-hit-est",
            "-i", fasta_file,
            "-o", cd_hit_file,
            "-c", str(sim),
            "-n", str(word_size),
            "-M", "16000",
            "-T", "8",
            "-d", "0"
        ]

        try:
            subprocess.run(cmd, check=True)
            logger.log_message("CD‑HIT‑EST finished successfully.")
        except subprocess.CalledProcessError as e:
            logger.log_message(f"CD‑HIT‑EST exited with a non‑zero status: {e}")
    else:
        logger.log_message(f"'{cd_hit_file}' already exists.")

    return cd_hit_file, f"{cd_hit_file}.clstr"

def process_clusters(cd_hit_clstr_file):
    """
        Processes the CD-HIT clustering output file and extracts relevant information to create a DataFrame.

        Args:
        - cd_hit_clstr_file (str): The path to the CD-HIT .clstr file.

        Returns:
        - DataFrame: A DataFrame containing parsed cluster data with columns for cluster, entity, representativeness,
                      sequence similarity, and other related metadata.
        """
    with open(cd_hit_clstr_file, 'r') as file:
        raw_data = file.readlines()

    parsed_data = []
    current_cluster = None

    for line in raw_data:
        line = line.strip()
        if line.startswith(">Cluster"):
            current_cluster = line.split(" ")[1]  # cluster number
        elif line:
            parts = line.split()

            entity = parts[2].split(">")[1]
            is_representative = "*" in line
            similarity = parts[-1].strip("%*") if "%" in parts[-1] else None
            parsed_data.append([current_cluster, entity, is_representative, similarity])

    df = pd.DataFrame(parsed_data, columns=["cluster", "entity", "is_representative", "sequence_similarity"])
    df['entity'] = df['entity'].str.replace('...', '', regex=False)
    df[['row index', 'species', 'chromosomeId', 'origin', 'label', '_in_test']] = df['entity'].str.split('_', expand=True)
    df['label'] = df['label'].astype(int)
    return df

def balance_dataset(logger, df):
    """
    Balances the dataset by ensuring equal numbers of label 0 and 1 within each '_in_test' group.
    """
    # get the minimum counts of each label within each '_in_test' group
    label_counts = df.groupby(['_in_test', 'label']).size().unstack(fill_value=0)
    label_counts['difference'] = label_counts.max(axis=1) - label_counts.min(axis=1)
    logger.log_message(f"Label counts and difference per _in_test group:\n{label_counts}")

    label_columns = label_counts.columns.drop('difference')
    min_counts = label_counts[label_columns].min(axis=1)

    logger.log_message(f"Minimum counts per _in_test group:\n{min_counts}")

    # sample the dataset to balance the labels within each '_in_test' group
    df_balanced =  df.groupby('_in_test').apply(
        lambda group: pd.concat([
            group[group['label'] == 0].sample(n=min_counts[group.name], random_state=64546),
            group[group['label'] == 1].sample(n=min_counts[group.name], random_state=64546)
        ])
    ).reset_index(drop=True)

    dist_before = df.groupby(['label', '_in_test']).size()
    dist_after = df_balanced.groupby(['label', '_in_test']).size()

    logger.log_message(
        f"Balanced dataset created. Removed {len(df) - len(df_balanced)} samples.\n"
        f"Original representative distribution:\n{dist_before}\n"
        f"Balanced representative distribution:\n{dist_after}\n"
        f"Final counts per chromosome:\n{df_balanced.groupby('chromosomeId').size()}"
    )
    return df_balanced

def get_non_similar_rows(logger, df, group_by_clause):
    """
        Selects one representative sequence per group (as defined by `group_by_clause`), prioritizing inclusion in the train set
        to maximize training size.
        Balances the dataset by ensuring equal numbers of opposite labels (0 and 1) within each '_in_test' group.

        Args:
        - logger: Logger object for logging messages.
        - df (DataFrame): The dataframe containing sequences and labels.
        - group_by_clause (str): The column name to group by (e.g., 'species', 'chromosomeId').

        Returns:
        - DataFrame: The balanced dataframe with representative sequences.
    """
    # select representative sequences for each group by prioritizing rows with '_in_test=False'
    df_representatives = df.groupby(group_by_clause, as_index=False).apply(
        lambda group: group.sort_values(by='_in_test').iloc[0]
    ).reset_index(drop=True)

    logger.log_message(f"{len(df) - len(df_representatives)} samples from '{group_by_clause}' group have been removed"
                       f"\nBalancing dataset...")

    # Balance the dataset by ensuring equal numbers of label 0 and 1 within each '_in_test' group
    return balance_dataset(logger, df_representatives)


def handle_inter_train_test_similarity(logger, df, sim, word_size, temp_dir):
    """
    This method performs sequence similarity clustering using CD-HIT,
    and then remove similar sequences, to prioritize deleting them from test split (_in_test) to maximize training size.

    Args:
    - logger: Logger object for logging messages.
    - df (DataFrame): Dataset.
    - sim (float): The sequence similarity threshold for CD-HIT.
    - word_size (int): The word size parameter for CD-HIT.
    - temp_dir (str): The temporary directory for storing intermediate CD-hit files.

    Returns:
    - non_sim_df (DataFrame): The processed non-similar dataset.
    """
    logger.log_message(f"Original size: {df['_in_test'].value_counts()}")

    df["row index"] = df.index + 1
    df['chromosomeId'] = df['chromosomeId'].str.replace('_', '-')
    df['entity'] = df.apply(
        lambda
            row: f"{row['row index']}_{row['species']}_{row['chromosomeId']}_{row['origin']}_{row['label']}_{row['_in_test']}",
        axis=1
    )

    # create fasta file abased on given data frames and run CD-HIT
    fasta_file = create_fasta(logger, df, temp_dir)
    cd_hit_file, cd_hit_clstr_file = run_cd_hit(logger, fasta_file, sim, word_size, temp_dir)

    # process CD-HIT clusters
    processed_df = process_clusters(cd_hit_clstr_file)
    file_name_without_ext = os.path.splitext(os.path.basename(cd_hit_file))[0]
    processed_df.to_csv(os.path.join(temp_dir, f'{file_name_without_ext}.csv'), index=False)

    # get balanced non-similar rows
    non_sim_df = get_non_similar_rows(logger, processed_df, 'cluster')
    non_sim_df = non_sim_df.drop(columns=['sequence_similarity', 'cluster'])

    # merge with original dataframes
    non_sim_df = non_sim_df.merge(df, on='entity', suffixes=('', '_duplicate'))

    # remove duplicate columns and unnecessary ones
    non_sim_df = non_sim_df.loc[:, ~non_sim_df.columns.str.endswith('_duplicate')].drop(
        columns=['entity', 'row index', 'is_representative'])

    return non_sim_df

