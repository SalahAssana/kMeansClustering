# Multi-Dimension Implementations of k-means Clustering

This project contains Python scripts that demonstrate the k-means clustering algorithm applied to datasets of varying dimensions. This repository contains the code for the Medium article [k-means Clustering: Math Explained & Multi-Dimension Implementations](https://medium.com/@SalahAssana/k-means-clustering-math-explained-multi-dimension-implementations-85e202bc2ec0). The goal is to showcase how the algorithm works in 1D, 2D, 3D, and N-dimensional spaces. The primary language used in this repository is Python.

## Repository Contents

The repository includes the following Python scripts, each tailored to a specific dimensionality:

*   `kMeans1D.py`: An implementation of k-means clustering for one-dimensional data.
*   `kMeans2D.py`: An implementation of k-means clustering for two-dimensional data.
*   `kMeans3D.py`: An implementation of k-means clustering for three-dimensional data.
*   `kMeansND.py`: A generalized implementation of k-means clustering for N-dimensional data.

It also contains standard repository files like `.gitignore` and this `README.md`.

## Getting Started

To use these examples, you will need to have Python installed. You can clone this repository and run the scripts individually to see the k-means algorithm in action.

```bash
git clone https://github.com/SalahAssana/kMeansClustering.git

cd kMeansClustering

python -m venv path/to/venv

source path/to/venv/bin/activate

pip install -r requirements.txt
```

Then, you can run any of the example scripts:

```bash
python kMeans2D.py
```