Proximity VAE
==============================

Variational Representation Learning by using proximity loss functions

The data and saved models can be downloaded from: [resources](https://drive.google.com/drive/folders/16pw2gXLBwM2_oIimMyHg3l-Xy7Jmc8m4?usp=sharing)

Replicate the conda environment in the environment file using: `conda env create -f environment.yml`

Train the model using the following command:

```
allennlp train -s models/info_vae_snli_cosine_reg --include-package src \
    config/info_Vae_snli_cosine_reg.jsonnet
```

Can perform transfer using the following command:

```
python prox_vae.py transfer models/info_vae_snli_cosine_reg --templates_file \
    data/interim/snli_1.0/templates.tsv
```
