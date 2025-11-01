# Image Feature-Based Food Recommendation System

## 1. Introduction

This project implements a **Content-Based Image Retrieval (CBIR)** recommendation system. The system works by extracting feature vectors from food images using Deep Learning models and then applying a nearest neighbor search algorithm (e.g., FAISS/Nearest Neighbors) to find and suggest similar food items.

## 2. Key Features

* **Feature Extraction:** Utilizes pre-trained Deep Learning models to transform food images into numerical feature vectors.
* **Multi-Model Support:** The system is flexibly designed to allow testing and comparison of the performance across various state-of-the-art CNN architectures for feature extraction, including:
    * **EfficientNet**
    * **ResNet**
    * **MobileNet**
    * **OSNet** (or other integrable models)
* **Similarity Search:** Uses the high-performance **FAISS** (Facebook AI Similarity Search) library for fast and accurate retrieval of the most similar food images.
* **Recommendation:** Takes an input image and returns the $K$ most recommended/similar food images (e.g., $K=5$).

## 3. Technologies Used

* **Language:** Python
* **Deep Learning Framework:** `TensorFlow` / `Keras`
* **Feature Extraction:** Models from `Keras Applications` (e.g., EfficientNetB0, ResNet50, MobileNetV2).
* **Fast Search:** `FAISS` (Facebook AI Similarity Search)
* **Data/Image Processing:** `NumPy`, `PIL (Pillow)`, `Matplotlib`
* **Environment:** Jupyter Notebook (`.ipynb`)

## 4. Setup and Installation

1.  **Clone Repository:**
    ```bash
    git clone <YOUR_REPO_ADDRESS>
    cd <REPO_NAME>
    ```

2.  **Install Libraries:**
    You need to install the necessary libraries. Pay special attention to `faiss-cpu` (or `faiss-gpu` if you have a GPU).
    ```bash
    pip install numpy pandas tensorflow keras pillow matplotlib
    pip install faiss-cpu  # Or faiss-gpu
    ```

## 5. Usage Guide

1.  **Download Data:** Ensure you have downloaded and organized the food image dataset into the correct path required by the notebook.
2.  **Open Notebook:** Open the `assigment-recomment-food.ipynb` file using Jupyter Notebook or JupyterLab.
    ```bash
    jupyter notebook assigment-recomment-food.ipynb
    ```
3.  **Model Configuration:** Within the notebook, you can easily change the model variable (e.g., from `EfficientNetB0` to `ResNet50`) to test the performance of different architectures.
4.  **Run Code:** Execute the cells sequentially. The notebook will perform the following steps:
    * Load the pre-trained model.
    * Extract features from the entire training dataset.
    * Build the FAISS search index.
    * Run a query with a sample image and display the recommended food items.
