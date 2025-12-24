# Potato Disease Classification

This project uses a CNN model to classify potato leaves into three categories: Early Blight, Late Blight, or Healthy. It features a TensorFlow-based training pipeline and a FastAPI wrapper for real-time predictions. The system is designed to automate disease detection and improve crop management.

## Dataset Credits
The dataset used in this project is sourced from the **PlantVillage** dataset.  
Credits: [Arjun Tejaswi (Kaggle)](https://www.kaggle.com/arjuntejaswi/plant-village)

## How to Use

### 1. Setup Data
* Place the `PlantVillage.zip` file in the same directory as `training.ipynb`.

### 2. Training
* Open `training.ipynb` in Jupyter Notebook.
* Ensure the dataset is extracted or the path in the second cell points to the local folder.
* Run all cells to train the model.
* The generated model `potatoes.keras` file would be stored in the `models/` directory.

### 3. API Deployment
Navigate to the API folder and start the server using Uvicorn:
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0
```

Access the API at: http://localhost:8000

## Tech Stack
* **TensorFlow**: Model Training and Deep Learning
* **Python**: Core Language
* **FastAPI**: Web Framework for API deployment
* **NumPy**: Data Manipulation and Matrix operations
* **Matplotlib**: Visualization of training results
