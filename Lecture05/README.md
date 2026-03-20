# Deep Learning – Lecture 5  
## Recurrent Neural Networks, LSTMs, and Time Series Forecasting

This repository contains the material for **Lecture 5 of the Deep Learning course**.

In this session we move from spatial data (images) to **sequential data**, focusing on how deep learning models can capture **temporal dependencies**. We introduce **Recurrent Neural Networks (RNNs)** and their improved variants, particularly **Long Short-Term Memory (LSTM) networks**, in the context of **time series forecasting**.

The lecture emphasizes how models process sequences, how memory is handled across time steps, and why specialized architectures are needed for temporal data.

The session is accompanied by slides, example code, and a practical **time series forecasting task using bike demand data**.

---

## Topics Covered

### Convolutional Neural Networks (CNNs)

Neural network architectures designed specifically for image data.  
Key components include:

- convolutional layers  
- feature maps  
- receptive fields  
- pooling layers  
- hierarchical feature extraction  

Students will understand how CNNs leverage spatial structure and why they outperform fully connected networks for visual tasks.

---

### Recurrent Neural Networks (RNNs)

Neural network architectures designed to process **sequential data** by maintaining a hidden state over time.

Key concepts include:

- sequential data representation  
- hidden state dynamics  
- parameter sharing across time  
- unfolding through time (computational graph)  
- limitations of vanilla RNNs (vanishing/exploding gradients)  

Students will understand how RNNs model temporal dependencies and why they differ fundamentally from feedforward networks.

---

### Long Short-Term Memory (LSTM)

An advanced RNN architecture designed to address the limitations of standard RNNs.

Key components include:

- cell state and hidden state  
- input, forget, and output gates  
- controlled information flow  
- long-term dependency modeling  

Students will analyze how LSTMs enable stable gradient flow and effective learning over long sequences.

---

### Time Series Forecasting

Application of deep learning models to predict future values based on historical observations.

Topics include:

- univariate vs multivariate time series  
- sliding window approaches  
- sequence-to-one vs sequence-to-sequence prediction  
- train/validation splits for temporal data  
- evaluation metrics for forecasting  

Students will learn how to structure time series data for deep learning models.

---

### Training Considerations for Sequential Models

Practical aspects when training RNN-based models:

- sequence length selection  
- batching sequential data  
- normalization and scaling  
- teacher forcing (conceptual)  
- model stability and convergence  

---

# Notebook Exercise – Bike Demand Forecasting

The practical exercise for this lecture is a **time series forecasting task** where students predict **bike demand** based on historical data.

The dataset contains temporal information such as:

- hourly or daily demand  
- weather-related features  
- seasonal patterns  

Students will use this data to train recurrent models and evaluate their predictive performance.

### Exercise Notebooks

https://github.com/jdmartinev/SI7011-DeepLearning/tree/main/Lecture05/notebooks/excercise

---

## Exercise Objective

In this exercise students will build and train **recurrent neural networks using PyTorch** for time series forecasting.

Students will explore:

- preparing time series datasets using **sliding windows**  
- training **vanilla RNNs** for forecasting  
- implementing and comparing **LSTM models**  
- analyzing the effect of **sequence length and input features**  
- evaluating model performance using appropriate metrics (e.g., MAE, RMSE)  

The exercise encourages students to **experiment with different architectures and configurations**, understand the behavior of recurrent models, and develop intuition for modeling temporal data.

## Deliverable

Students must submit a **GitHub repository** containing the completed notebooks for this exercise.

### Requirements

The repository should include:

- The **executed notebooks** (with outputs visible)  
- Clear implementation of:
  - RNN models  
  - LSTM models  
- Experiments showing:
  - different sequence lengths and/or input configurations  
  - comparison between models  
- Proper organization and readability (clean code, comments, structure)

### Evaluation Criteria

Submissions will be evaluated based on:

- **Correct implementation** of RNN and LSTM architectures  
- **Experimental analysis** and comparison between approaches  
- **Code quality and organization**  
- **Reproducibility** (notebooks run without errors and outputs are included)  

Students are encouraged to iterate on their models and document their findings directly in the notebooks.
