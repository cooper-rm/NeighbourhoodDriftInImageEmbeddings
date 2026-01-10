### Capacity-Limited Failure in Approximate Nearest Neighbors Search on Image Embedding Spaces

## **Abstract**
Similarity search on image embeddings is a common practice for image retrieval in machine learning and pattern recognition systems. Approximate Nearest Neighbors (ANN) methods enable scalable similarity search on large datasets, often approaching sub-linear complexity. Yet, little empirical work has examined how ANN neighborhood geometry differs from that of exact k-nearest neighbors (k-NN) search as the neighborhood size increases under constrained search effort. This study quantifies how approximate neighborhood structure changes relative to exact k-NN search as k increases across three experimental conditions. Using multiple random subsets of 10000 images drawn from the STL-10 dataset, we compute ResNet-50 image embeddings, perform an exact $k$-NN search, and compare it to a Hierarchical Navigable Small World (HNSW) based ANN search under controlled hyperparameter regimes. Fidelity of neighborhood structure is evaluated using neighborhood overlap, average neighbor distance, normalized barycenter shift, and local intrinsic dimensionality (LID). Results show that exact $k$-NN and ANN search behave nearly identically when efSearch > k, but when the neighborhood size grows, and efSearch remains fixed, ANN search fails abruptly, exhibiting catastrophic divergence in neighbor distances at approximately k * 2.0 to 3.5 efSearch. Increasing index construction quality delays this failure, and scaling search-effort proportionally with neighborhood size (efSearch = a * k with a > 1.0) preserves neighborhood geometry across all evaluated metrics, including LID. These findings demonstrate that ANN search preserves neighborhood geometry when operating within search capacity, but abruptly fails when this capacity is exceeded. Documenting this behavior is relevant for scientific applications that approximate embedding spaces and provides practical guidance on when ANN search is interchangeable with exact $k$-NN and when geometric differences become nontrivial.


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
