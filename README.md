# Harnessing DNA Foundation Models for Cross-Species TFBS Prediction in Plants

![Overview of the fine-tuning framework for transcription factor binding sites (TFBS) prediction](images/overview.png)

## Overview

This project fine-tunes large pretrained DNA foundation models to predict transcription factor binding sites (TFBSs) in plant genomes.
We benchmark three foundation models:**DNABERT-2**, **AgroNT**, and **HyenaDNA**, against specialized methods like **DeepBind** and **BERT-TFBS** using DAP-seq data from *Arabidopsis thaliana* and *Sisymbrium irio*.
The project uses a unified pipeline, which covers three evaluation protocols:

1. **Cross-chromosome**: Leave-one-chromosome-out evaluation on *A. thaliana* (Sun2022). 
2. **Cross-dataset**: Train on Malley2016 AREB/ABF2 dataset, test on Sun2022 AREB/ABF2 dataset.
3. **Cross-species**: Train on one species, test on the other (Sun2022).

The **HyenaDNA** model achieves near state-of-the-art accuracy while training in minutes, demonstrating both predictive power and computational efficiency.

## Installation

To get started, follow these steps:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Maryam-Haghani/TFBS.git
   cd TFBS
   ```

2. **Create a conda environment**  
   ```bash
   conda create -n tfbs-env python=3.9
   conda activate tfbs-env
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   
4. **Navigate to the code directory**:
   ```bash
   cd ./TFBS/code
   ```

---

## 1. Data Preparation
Data utilized in this project are sourced from the [Ronan2016](https://pubmed.ncbi.nlm.nih.gov/27203113/) and [Sun2022](https://pubmed.ncbi.nlm.nih.gov/35501452/) studies. Download the data and place it in the `./inputs` directory inside the `./TFBS` project folder.

We used the `01-generate_samples.py` script to process raw data and generate positive and negative samples for transcription factor (TF) binding sites.
It takes as input a FASTA file of chromosome sequences for species genome and a CSV file containing peak regions.
Negative sample will be generated based on `neg_type` argument with default value of "shuffle".

### Usage
To run the script, use the following command:

```bash
python 01-generate_samples.py --fasta_file path/to/your.fasta --peak_file path/to/peaks.csv --output_file path/to/output.csv --neg_type shuffle --species SI/ATA --dataset Josey/Ronan
```

#### Arguments
- `--fasta_file`: Path to the genome FASTA file.
- `--peak_file`: Path to the CSV file containing peak data.
- `--species`: Species type (currently supporting *S. irio* and *A. thaliana*).
- `--dataset`: Origin of peak files.
- `--output_file`: Path to save the output CSV file to contain positive and negative samples.
- `--neg_type`: Method to generate negative samples (e.g., "dinuc_shuffle", "shuffle", "random").
- `--sliding_window`: Window around the peak region. Sample lengths vary based on the original peak length.
- `--fixed_length`:  If specified, generates samples of fixed length, ignoring the sliding window.

#### Example
##### For *A. thaliana* (ABF1-4) dataset:
```bash
python 01-generate_samples.py --fasta_file ../inputs/fastas/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa --peak_file ../inputs/peak_files/AtABFs_DAP-Seq_peaks.csv --species "At" --dataset Josey  --output_file ../inputs/AtABFs_shuffle_neg_stride_200.csv
```
##### For *S. irio* (ABF1-4) dataset:
```bash
python 01-generate_samples.py --fasta_file ../inputs/fastas/Si_sequence --peak_file ../inputs/peak_files/SiABFs_DAP-Seq_peaks.csv --species "Si" --dataset Josey  --output_file ../inputs/SiABFs_shuffle_neg_stride_200.csv  
```
### Output
This will generate positive and negative samples based on the given negative type generation, for the given species using the provided FASTA file and peak data, saving the results to `--output_file`.


## 2. Data Split
### Description

This script splits the data based on the provided configuration file.

### Usage
To run the script, use the following command:

```bash
python 02-split_data.py --config_file [config_path]
```

#### Arguments
- `--config_file`: Path to the configuration YAML file, which contains settings for the data split (file paths, random seed, etc.).

Data split configurations are located in the `/configs/data_split` directory.

#### Cross-configs
- `cross_chromosome_config.yml`: Use `test_id` values between **1** and **5**.
- `cross_species_config.yml`: Use `test_id` as either **At** or **Si**.
- `cross_dataset_config.yml`: Use provided dataset identifiers.

#### Example
If you have a configuration file located at `../configs/data_split/cross-species-config.yml`, run:

```bash
python 02-split_data.py --config_file ../configs/data_split/cross-chromosome-config.yml
```

### Output
Data splits are stored in `<config.split_dir>/<config.name>`.


## 3. Training / Fine-tuning Models on a Split
### Description
This script trains and evaluates machine learning models, with a grid search over hyperparameters like batch size, learning rate, and weight decay.
It integrates with [Weights & Biases (wandb)](https://wandb.ai) for experiment tracking and visualization.

### Usage
To run the script, use the following command:

```bash
python 03-train.py --train_config_file [train_config_path] --split_config_file [data_config_path]
```

#### Arguments
- `--train_config_file`: Path to the training configuration YAML file.
- `--split_config_file`: Path to the data split configuration YAML file.

#### Example
```bash
python 03-train.py --train_config_file ../configs/train/HeynaDNA-config.yml --split_config_file ../configs/data_split/cross-species-config.yml
```

### Output
Trained models and logs are saved in `<config.output_dir>/<split_config.name>/`

## 4. Prediction
### Description
This script generates predictions from saved pre-trained models for a dataframe or a genome sequence.

### Usage
To run the script, use the following command:

```bash
python 04-predict.py --config_file [config_path] --mode= [mode]
```

#### Arguments

- `--config_file`: The path to the configuration YAML file.
--mode: Specifies the data mode. Use "df" when the input data is in a dataframe format with `sequence` and `label` columns. Use "genome" when the input is a sequence, and the script predicts whether any fragment of the sequence, with a length defined by `config.window_size` and a stride defined by `config.stride`, is a binding site.

### Example
#### mode = "df"
```bash
python 04-predict.py --config_file ../configs/predict/HeynaDNA-config.yml --mode df
```
##### Output
A directory named `<config.input_dir without extension>` will be created inside the `Predictions` folder, located within the saved model directory (`config.saved_model_dir`).
Inside this directory, a CSV file containing predictions will be saved, named after each model, corresponding to the models saved in the model directory.
Additionally, a `prediction_result.csv` file will be generated, summarizing performance metrics for all models.

#### mode = "genome"
```bash
python 04-predict.py --config_file ../configs/genome_predict/HeynaDNA-config.yml --mode genome
```
##### Output
A directory named `<config.input_dir without extension>` will be created inside the `Predictions` folder within the saved model directory (`config.saved_model_dir`).
In this directory, a plot will be saved for each model, with the filename following the template: `peak-[model_name]-window_size_[config.window_size]-stride_[config.stride].png`.

## 5. Embedding Generation and Visualization
This script extracts embeddings from a pre-trained or saved model for a data split, and visualizes them using PCA, T-SNE, or UMAP.
The script supports the use of pre-trained models or saved models, based on the configuration provided (`use_pretrained_model`).

Each model has its own config in `/configs/embedding`.

### Usage

To run the script, use the following command:

```bash
python 05-get_embedding.py --split_config_file [split_config_path] --embed_config_file [embed_config_path]
```

#### Arguments
- `--split_config_file`: The path to the data split configuration YAML file.
- `--embed_config_file`: The path to the embedding configuration YAML file.

#### Example
```bash
python 05-get_embedding.py --embed_config_file ../configs/embedding/HeynaDNA-config.yml --split_config_file ../configs/data_split/cross-species-config.yml
```

### Output
Embeddings are saved in `.pt` files, and visualizations are generated in `plots` subdirectory.

## Requirements

- Python 3.9+  
- PyTorch 1.12+  
- Transformers 4.20+  
- scikit-learn 1.0+  
- pandas, numpy, matplotlib  

See **requirements.txt** for exact versions.

---

## Citation

If you use this code or data, please cite:

> Haghani M., Dhulipalla K.V., Li S. “Harnessing DNA Foundation Models for Cross-Species Transcription Factor Binding Site Prediction in Plant Genomes.” *bioRxiv* (2025).  

```bibtex
@article{Haghani2025TFBS,
  title   = {Harnessing DNA Foundation Models for Cross-Species Transcription Factor Binding Site Prediction in Plant Genomes},
  author  = {Haghani, Maryam and Dhulipalla, Krishna Vamsi and Li, Song},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/XXXXXX}
}
```
---

## Contact

For questions or issues, please open an issue on GitHub or contact **haghani@vt.edu**.
