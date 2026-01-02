### Empirical Analysis of Neighborhood Drift in Image Embeddings: Approximate vs. Exact Nearest Neighbors Search

## **Abstract**
Similarity search on image embeddings is a common practice for image retrieval in machine learning and pattern recognition systems. Approximate Nearest Neighbors (ANN) methods enable scalable similarity search on large datasets, often approaching sub-linear complexity. Yet little empirical work on measuring how ANN neighborhood geometry differs from exact $k$-nearest neighbors ($k$-NN) search, for varying neighborhood sizes. This study quantifies how ANN neighborhood structure changes relative to exact $k$-NN search as $k$ increases. Using 10,000 images from the STL-10 dataset, we compute ResNet-50 image embeddings, perform an exact $k$-NN search, and perform an approximate search with an HNSW-based ANN index under fixed hyperparameters. Fidelity of Neighborhood structure is evaluated using neighborhood overlap, average neighbor distance, barycenter shift, and local intrinsic dimensionality (LID). Results show that exact $k$-NN search and ANN search are nearly identical for small $k$; below the region ranging from 50 to 90. Beyond this region, ANN search gradually diverges: neighborhood overlap declines, average neighbor distance increases, barycenters shift and LID is constantly underestimated. These findings demonstrate that ANN search preserves core neighborhood geometry but increasingly deforms periphery regions of neighborhoods. Documenting this behavior is relevant for scientific applications that are sensitive to manifold topology such as clustering, density estimation, and dimensionality analysis, and quantifying the deformation of neighborhoods provides useful insight into when ANN search is interchangeable with exact k-NN and when geometric differences become nontrivial.


# Environment Setup (Miniconda)

This project uses a dedicated conda environment for reproducibility.  

Follow the steps below to install Miniconda, create the environment, and install all dependencies.

---

## 1. Install Miniconda
Download the installer for your OS:  
https://docs.conda.io/en/latest/miniconda.html  
Follow the installation instructions, then restart your terminal.

---

## 2. Create and activate the environment
```bash
conda create -n drift-exp python=3.10 -y
conda activate drift-exp
```

## 3. Install dependancies 
```bash
pip install requirements.txt
```
