# Single-Cell-Perturbations
Kaggle competition project--Single cell perturbations
## Dataset

### Details

selected 144 compounds(2--dabrfenib and belinostat as postive controls and DMSO as negative control) from LINCS to PBMCs from 3 donors
<img width="1565" alt="inbox%2F4308072%2Fdbce2b5e0a2b8b9691502b31c5feb5b6%2FScreenshot 2023-08-25 at 6 20 53 PM" src="https://github.com/Kainan-Liu/Single-Cell-Perturbations/assets/146005327/72f3a46b-12c4-463d-a837-9dcd45e3409f">


The plate contains 96 wells, each well contains PBMCs from a donor(each well contains cells belonging to all cell types), 72 wells--compound, 16--positive controls, 8--negative controls, The full dataset comprises 2 different compound plates per donor for a total of 6 plates and 350 cells per well

Why introduce two positive controls and negative controls? One reason is that when we cell multiplexing(pool all samples in each row into a single pool for sequencing), two positive controls and one negative control in each row of the plate is to allow us to **account for this source** of noise when we calculate differential expression.

阳性对照组化合物对细胞的gene expression影响较大，而阴性对照组是其他化合物的solvent?主要的作用是calculate differential gene expression时的对照(reference)-- there is no DE data for the DMSO sample, because it is the negative control. All DE output is calculated in reference to the DMSO, i.e. the DE analysis asks "how confident am I that each gene increased or decreased relative to DMSO due to the compound treatment".

### Data splits

- Training dataset: All compounds in T, NK cells and 10% of the compounds in B and Myeloid cells
- Testing dataset: randomly chosen compounds in B and Myeloid cells
- 训练数据集囊括了所有测试数据集上的compound/cell_type pair

### Main dataset

- de_train.parquet

  614 cells, 18211 genes(The first 5 columns are cell types/compound pair and Boolean indicator of control)

- adata_train.parquet

  adopt  different format--COO sparse--array format, other fileds: obs_id...

## Tasks Descriprion

### Overview

Modelling differential expression, predict the gene expression differential data in reference to the negative controls(DMSO)

### Evaluation Metric

Mean Rowwise Root Mean Squared Error(MRRMSE)

![image-20231016214521034](https://github.com/Kainan-Liu/Single-Cell-Perturbations/assets/146005327/04d5ad6f-60ae-45be-8ab7-62fe230b7434)


i: represent the cells, and j:  represent the genes

Several methods have been developed for drug perturbation prediction, most of which are variations on the **autoencoder architecture (Dr.VAE, scGEN, and ChemCPA).**
