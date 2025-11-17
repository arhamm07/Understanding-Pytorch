# Understanding PyTorch

 A collection of small, self-contained PyTorch examples that show how to build end-to-end deep learning pipelines on different kinds of data:

 - Tabular data
 - Images (from scratch and with transfer learning)
 - Audio
 - Text (with BERT)

 Each example lives in its own Jupyter notebook and walks through data loading, preprocessing, model definition, training, and evaluation.

 ## Repository structure

 - `1_Tabular_Data_Classification/`
   - `rice_type_classification.ipynb` – Binary classification of rice varieties from tabular features.

 - `2_Image_Classification/`
   - `animal_image_classification.ipynb` – Animal faces image classification using a Kaggle dataset and a custom `Dataset` / `DataLoader` pipeline.

 - `3_Classification_using_pretrained_model/`
   - `Classification_with_transfer_learning.ipynb` – Image classification with transfer learning from a pretrained CNN in `torchvision.models` on a bean leaf lesion dataset.

 - `4_Audio_Classification/`
   - `audio_classification.ipynb` – Audio classification with PyTorch (see the notebook for dataset and feature-extraction details).

 - `5_Text_Classification/`
   - `Text_CLassification.ipynb` – Text classification on news headlines (sarcasm detection) using a BERT encoder.

 ## Prerequisites

 You will need a Python environment with:

 - Python 3.8+ (or similar)
 - [PyTorch](https://pytorch.org/)
 - `torchvision`
 - Jupyter Notebook or JupyterLab (or VS Code with the Python extension)
 - Common data-science libraries:
   - `pandas`, `numpy`, `matplotlib`, `scikit-learn`
 - Notebook-specific extras:
   - `opendatasets` (for downloading some Kaggle datasets)
   - `transformers` (for the BERT-based text classification notebook)

You can install the core dependencies, for example, with:

```bash
pip install torch torchvision
pip install jupyter pandas numpy matplotlib scikit-learn opendatasets transformers
```

> **Note**: For GPU support, install the PyTorch version that matches your CUDA / system configuration as described on the official PyTorch website.

 ## Running the notebooks locally

 1. Clone the repository:

    ```bash
    git clone https://github.com/arhamm07/Understanding-Pytorch.git
    cd Understanding-Pytorch
    ```

 2. Start Jupyter:

    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```

 3. Open any of the notebooks (for example `1_Tabular_Data_Classification/rice_type_classification.ipynb`) and run the cells from top to bottom.

 ## Datasets

 Some notebooks expect datasets from Kaggle and use `opendatasets` to download them:

 - `2_Image_Classification/animal_image_classification.ipynb` – uses an *Animal Faces* dataset from Kaggle.
 - `3_Classification_using_pretrained_model/Classification_with_transfer_learning.ipynb` – uses a bean leaf lesion classification dataset from Kaggle.
 - `5_Text_Classification/Text_CLassification.ipynb` – uses a news headlines sarcasm detection dataset.

 To use `opendatasets` with Kaggle:

 1. Create a Kaggle account and generate an API token.
 2. Place `kaggle.json` in the default location (`~/.kaggle/kaggle.json`) or configure credentials as required by `opendatasets`.
 3. Run the download cell in the corresponding notebook, or download datasets manually from Kaggle and update the file paths in the notebook.

 ## Purpose

 This repository is primarily for learning and experimentation with PyTorch:

 - Understanding how to implement custom `Dataset` and `DataLoader` classes.
 - Practicing model design, training loops, and evaluation.
 - Seeing how the same deep learning framework applies to tabular, image, audio, and text data.

 Feel free to use these notebooks as a starting point for your own projects or as reference material when revising PyTorch concepts.