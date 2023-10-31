# stMGATF
Revealing Tissue Heterogeneity and Spatial Dark Genes from Spatially Resolved Transcriptomics by Multi-view Graph Networks

## Requirement
- Reference

  https://docs.anaconda.com/anaconda/install/index.html

  https://pytorch.org/

- python==3.6.12

- numpy>=1.19.2

- pandas>=1.1.5

- scipy>=1.5.2

- scikit-learn>=0.23.2

- torch>=1.6.0

- tqdm>=4.55.0

- scanpy>=1.6.0

- json==2.0.9

- stlearn

- sklearn

- matplotlib==3.5.2

- glob2

- anndata==0.8.0

- argparse==1.1

- pathlib

  

## Usage
#### Clone this repo.
```python
git clone https://github.com/liying-1028/stMGATF.git
cd stMGATF
```

#### Code description
- ```step1_Preprcessing.py```: Using an autoencoder framework to extract feature matrix
- ``extract_dna.py``: Learning visual features by SimCLRv2
- ```Cha_h2.m```: Processing spatial location information to construct SLG
- ```TEST_GAG.m```: Constructing the association matrix to build GAG
- ```step2_train.py```: Learning low-dimensional representations of each view by EGAT
- ```step3_GAM.py```: Global attention mechanism
- ```visualization.R```:Clustering visualization

#### Example 
Take the dataset "DLPFC_151673" as an example

- Step 1: preprocessing running pipeline:

  ```
  step1_Preprcessing.py
  extract_dna.py
  Cha_h2.m
  TEST_GAG.m
  ```
- Step 2: EGAT running pipeline:

  ```
  step2_train.py
  ```

- Step 3: GAM running pipeline:

  ```
  step3_GAM.py
  ```

  # Note: 

  You can find the data in 'stMGATF_test_data.txt'.

  The folder named 'DLPFC_151673' contains the raw data of slice 151673.

  To reduce your time, we have uploaded our preprocessed data into the folder 'output'. You can perform the corresponding steps selectively.

  You can modify parameters in `utils.py`.To reproduce the result, you should use the default parameters.

  # Citation

Li Y, Lu Y, Kang C, Li P, Chen L. Revealing Tissue Heterogeneity and Spatial Dark Genes from Spatially Resolved Transcriptomics by Multiview Graph Networks. Research 2023;6:Article0228 https://doi.org/10.34133/research.0228
