# Harnessing DNA Foundation Models for Cross-Species TFBS Prediction in Plants

![Overview of the fine-tuning framework for transcription factor binding sites (TFBS) prediction](images/overview.png)

## Overview

This project fine-tunes large pretrained DNA foundation models to predict transcription factor binding sites (TFBSs) in plant genomes. We benchmark three foundation models (DNABERT-2, AgroNT, HyenaDNA) against specialized methods (DeepBind, BERT-TFBS) on DAP-seq data from *Arabidopsis thaliana* and *Sisymbrium irio*. Our unified pipeline covers three evaluation protocols:

1. **Cross-chromosome**: Leave-one-chromosome-out on *A. thaliana* (Sun2022).  
2. **Cross-dataset**: Train on Malley2016 AREB/ABF2, test on Sun2022 AREB/ABF2.  
3. **Cross-species**: Train on one species, test on the other (Sun2022).

HyenaDNA achieves near state-of-the-art accuracy while training in minutes (versus hours), demonstrating both predictive power and computational efficiency.

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Maryam-Haghani/TFBS.git
   cd TFBS
   ```

2. **Create a conda/env virtual environment**  
   ```bash
   conda create -n tfbs-env python=3.9
   conda activate tfbs-env
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
---

## 1. Data Preparation

1. **Download raw DAP-seq data** into `data/raw/`:  
   - **Sun2022** AREB/ABF1–4 peaks for *A. thaliana* & *S. irio*  
   - **Malley2016** AREB/ABF2 peaks for *A. thaliana*  
2. **Run preprocessing** (pad/truncate, negative sampling, 5-fold splits):  
   ```bash
   python src/preprocess.py      --input-dir data/raw/      --output-dir data/processed/      --protocol cross_chromosome      --chromosomes 1 2 3 4 5      --lengths 264
   ```
   Repeat for `cross_dataset` (length = 201) and `cross_species` (length = 264).

## 2. Data Split

All data split configurations are stored in the `/configs/data_split` directory.

- **Cross configs**:
  - `cross_cromosome_config.yml`
    - `test_id`: choose an integer from **1** to **5**.
  - `cross_species_config.yml`
    - `test_id`: choose one of **At**, **Si**.
  - `cross_dataset_config.yml`
    - `test_id`: use the provided identifiers for your datasets.

To adjust a cross configuration:
1. Open the desired `cross_*_config.yml` file.
2. Set `test_id` to one of the valid values.

### Generating the Splits

Run the split script:

```bash
python 02-split_data.py   --config /configs/data_split/<your_config>.yml
```

- The script reads:
  - `config.split_dir`: root output directory.
  - `config.name`: name of the split.
  - `dataset_path`: folder containing the source CSV files.
- Splits are created in:
  ```
  <config.split_dir>/<config.name>/
  ```

## 3. Training / Fine-tuning Models on a Split

Each model has its own config in `/configs/train`. You should use the model config along with the specified data_split config.

Run the training script to fine-tune each model end-to-end with a two-neuron classification head:

```bash
python 03-train.py   --config_file [train_config_path] --split_config_file [data_split_config-path]
```

Trained models and logs are saved in:
```
<config.output_dir>/<split_config.name>/
```

## 4. Prediction
Each model has its own config in `/configs/predict`.

Run the test script to predict transcription factor binding site for a test dataset (`config.dataset_dir`) based on the saved model (`config.saved_model_dir`):


```bash
python 04-predict.py --config_file [config_path]
```

Script will output CSVs in a folder named as dataset name (csv file without .csv extension) in prediction folder and figures in `plot` directory in the root directory of `config.model.saved_model_dir`.


## 5. Embedding
Each model has its own config in `/configs/embedding`.

Run the embedding script to get embeddings using the specified model (`config.model`) for a dataset (`config.dataset_dir`) based on either
### Using the original pretrained model (`config.model.model_name`)
```bash
python 05-get_embedding.py --config_file [train_config_path]
```
In the config file, user should provide the dataset path in `config.dataset_dir` directory.
Script will save a pt file containing sequences, their embeddings and their true labels in  EMBEDDINGS-`config.dataset_dir` directory with the name of model .pt

### b Using saved model (`config.model.saved_model_dir`)
```bash
python 05-get_embedding.py --config_file [config_path] --split_config_file [data_split_config-path]
```
In this case, the script will get dataset to generate embeddings for based on `split_config_file` file
Script will save a pt file containing sequences, their embeddings and their true labels in the `config.model.saved_model_dir`/`Embedding` directory for all models inside the saved_model_dir

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

[//]: # (## License)

[//]: # ()
[//]: # (This project is released under the [MIT License]&#40;LICENSE&#41;.)

---

## Contact

For questions or issues, please open an issue on GitHub or contact **haghani@vt.edu**.
