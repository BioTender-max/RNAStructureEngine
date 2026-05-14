# RNAStructureEngine

**RNA Secondary Structure Prediction and Analysis Pipeline**

A pure-Python pipeline for RNA secondary structure prediction using the Nussinov algorithm with SHAPE reactivity integration.

## Features
- Minimum free energy (MFE) folding (Nussinov dynamic programming)
- Base-pair probability matrix (partition function approximation)
- SHAPE reactivity integration (constrained folding)
- Structural conservation scoring across homologs
- Structure-function correlation (UTR structure vs expression)

## Results
- 200 RNA sequences (100-500 nt)
- Mean MFE score: 29.18 base pairs
- Mean SHAPE-constrained MFE: 28.64
- MFE-Expression correlation: r=-0.050
- Mean conservation score: -3.72

## Usage
```bash
pip install numpy scipy matplotlib
python rna_structure_engine.py
```

## Tags
`rna-structure` `secondary-structure` `mfe-folding` `shape-reactivity` `pseudoknot` `nussinov`
