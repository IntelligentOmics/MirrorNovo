# MirrorNovo
MirrorNovo is a de novo sequencing model based on the two mirror proteases, 
the use of LysargiNase and Trypsin for protein digestion provides complementary ion types for MS/MS analysis,
resulting in a higher coverage of fragment ions.
# The usage of our code
## Preparation:
```
conda env create -n mirrornovo_env -f requirements.yml
```


## The params.cfg used in MirrorNovo

- `knapsack`: Path to the knapsack matrix file automatically generated based on the amino acid list. The generation may take a long time on first run.

- `train_try_spectrum_path`: Path to the MS/MS spectra of the trypsin training set.

- `valid_try_spectrum_path`: Path to the MS/MS spectra of the trypsin validation set.

- `train_lys_spectrum_path`: Path to the MS/MS spectra of the LysargiNase training set.

- `valid_lys_spectrum_path`: Path to the MS/MS spectra of the LysargiNase validation set.

- `train_feature_path`: Path to the CSV file containing metadata of the training spectra, such as mirror spectrum pairs and target amino acid sequences.

- `valid_feature_path`: Path to the CSV file containing metadata of the validation spectra, such as mirror spectrum pairs and target amino acid sequences.

- `denovo_input_spectrum_file`: Path to the MS/MS spectra of the trypsin test set.

- `denovo_input_mirror_spectrum_file`: Path to the MS/MS spectra of the LysargiNase test set.

- `denovo_input_feature_file`: Path to the CSV file containing metadata of the test spectra, such as mirror spectrum pairs.

- `denovo_output_file`: Output path for the de novo sequencing results.