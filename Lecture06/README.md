# Deep Learning – Lecture 6  
## Transformers and Neural Architectures for Text Classification

This repository contains the material for **Lecture 6 of the Deep Learning course**.

In this session we move from sequential models based on recurrence to **attention-based architectures**, focusing on how modern models process and understand text data. We introduce the **Transformer architecture**, which has become the foundation of state-of-the-art models in Natural Language Processing (NLP).

We also explore **neural architectures for text classification**, ranging from classical approaches to modern transformer-based methods.

The lecture is accompanied by slides, example code, and a practical **text classification task using deep learning models**.

---

## Topics Covered

### Text Representation for Deep Learning

Before introducing architectures, we discuss how text is converted into numerical representations.

Key concepts include:

- tokenization  
- vocabulary and indexing  
- padding and truncation  
- word embeddings (learned vs pre-trained)  

Students will understand how raw text becomes suitable input for neural networks.

---

### Recurrent and Convolutional Models for Text

Overview of early deep learning approaches for NLP tasks.

Topics include:

- RNNs and LSTMs for sequence modeling  
- CNNs for text classification  
- limitations of sequential processing  

This section provides context for the transition to transformer-based models.

---

### Attention Mechanism

A key idea that enables models to focus on relevant parts of a sequence.

Key concepts include:

- query, key, and value representations  
- attention scores and weighting  
- contextual representations  

Students will understand how attention improves over fixed-context representations.

---

### Transformer Architecture

A modern architecture that replaces recurrence with attention mechanisms.

Key components include:

- self-attention  
- multi-head attention  
- positional encoding  
- feed-forward networks  
- encoder architecture  

Students will understand how transformers process sequences in parallel and capture long-range dependencies.

---

### Pre-trained Language Models

Introduction to models trained on large corpora and adapted to downstream tasks.

Topics include:

- representation learning in NLP  
- fine-tuning vs feature extraction  
- examples of transformer-based models  

Students will understand why pre-trained models dominate modern NLP.

---

### Text Classification with Deep Learning

Application of neural models to classify text into categories.

Topics include:

- sequence classification setup  
- model architectures for classification  
- evaluation metrics (accuracy, F1-score)  
- handling class imbalance  

Students will learn how to design and evaluate text classification systems.

---

# Notebook Exercise – Text Classification

The practical exercise for this lecture is a **text classification task**, where students will build models to classify text into predefined categories.

The dataset may include:

- sentences, reviews, or short documents  
- categorical labels (multi-class or binary)  

Students will implement and compare different deep learning approaches for text classification.

---

## Exercise Objective

In this exercise students will build and train **deep learning models for text classification**.

Students will explore:

- preprocessing text data (tokenization, padding)  
- training **baseline models** (e.g., simple neural networks or RNNs)  
- implementing **attention-based or transformer-based models**  
- fine-tuning **pre-trained transformer models**  
- evaluating model performance using appropriate metrics  

The exercise encourages students to **compare architectures**, understand trade-offs, and explore modern NLP pipelines.

---

## Deliverable

Students must submit a **GitHub repository** containing the completed notebooks for this exercise.

### Requirements

The repository should include:

- The **executed notebooks** (with outputs visible)  
- Clear implementation of:
  - at least one baseline model (e.g., RNN/CNN)  
  - at least one transformer-based model  
- Experiments showing:
  - comparison between models  
  - impact of preprocessing or hyperparameters  
- Proper organization and readability (clean code, comments, structure)

### Evaluation Criteria

Submissions will be evaluated based on:

- **Correct implementation** of models  
- **Use of transformer-based approaches**  
- **Experimental analysis and comparison**  
- **Code quality and organization**  
- **Reproducibility** (notebooks run without errors and outputs are included)  

Students are encouraged to iterate on their models and document their findings directly in the notebooks.
