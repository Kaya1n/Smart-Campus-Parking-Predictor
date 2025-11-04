# 🎓 SmartCampus Parking Predictor: GDG AI Bootcamp Final Project

This project implements a Deep Neural Network (DNN) to predict university parking lot occupancy, utilizing time-series and academic features. This was developed as the final project for the **GDG AI Bootcamp**.

## Key Features and Techniques

-   **Model:** Sequential Deep Neural Network (DNN) using Keras/TensorFlow.
-   **Objective:** Regression (predicting a continuous value: Occupancy Rate from 0.0 to 1.0).
-   **Data Foundation:** Utilized **patterned synthetic data** (simulated 5,000 records) to reflect real-world parking behavior across different academic schedules, days, and times. This showcases strong **Feature Engineering** skills when real data is unavailable.
-   **Stability:** Implemented **LeakyReLU** activation functions in hidden layers and **Gradient Clipping** to ensure stable training and overcome common 'nan' loss errors.
-   **Preprocessing:** Used `OneHotEncoder` for categorical data (Day, Weather) and `StandardScaler` for numeric features (Temperature, Time).

## 📊 Model Performance

The model was rigorously tested to ensure high predictive reliability for campus facility management.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Model Accuracy (Based on MAE)** | **98.81%** | The model's predictions are highly reliable. |
| **Test Mean Absolute Error (MAE)** | **0.0119** | Average error is only **1.19%** of the true occupancy rate. |
| **Scenario Prediction (Peak Stress)** | **42.12%** | Predicted occupancy during a high-enrollment, exam-week peak hour. |

## Files in this Repository

-   `parking_predictor_unified.py`: Contains the complete Python code (Loading, Preprocessing, DNN Training, and Evaluation) in a single runnable script.
-   `parking_data_5000_records.csv`: The synthetic dataset used for training and testing.

## Prerequisites

-   Python 3.8+
-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `tensorflow`

## Running the Model

1.  Clone this repository.
2.  Ensure `parking_data_5000_records.csv` is in the same directory as the Python script.
3.  Run the script from your terminal:
    ```bash
    python parking_predictor_unified.py
    ```

The script will output the training progress, evaluation metrics, and the final scenario prediction.
