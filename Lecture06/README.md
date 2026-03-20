# Deep Learning – Lecture 6  
## Transformers and Neural Architectures for Text Classification

This repository contains the material for **Lecture 6 of the Deep Learning course**.

In this session we move from sequential models based on recurrence to **attention-based architectures**, focusing on how modern models process and understand text data. We introduce the **Transformer architecture**, which has become the foundation of state-of-the-art models in Natural Language Processing (NLP).

We also explore **neural architectures for text classification**, progressing from data processing pipelines to **pre-trained transformer models and efficient fine-tuning techniques**.

The lecture is accompanied by slides, example code, and a practical **text classification pipeline using the TweetEval dataset**.

---

## Topics Covered

### Text Representation for Deep Learning

Before introducing architectures, we discuss how text is converted into numerical representations.

Key concepts include:

- tokenization  
- vocabulary and indexing  
- padding and truncation  
- word embeddings (learned vs pre-trained)  

---

### From Sequential Models to Attention

Overview of earlier approaches and their limitations:

- RNNs and LSTMs for sequence modeling  
- CNNs for text classification  
- limitations of sequential processing  

This motivates the transition to attention-based models.

---

### Attention Mechanism

A key idea that enables models to focus on relevant parts of a sequence.

Key concepts include:

- query, key, and value representations  
- attention scores and weighting  
- contextual representations  

---

### Transformer Architecture

A modern architecture that replaces recurrence with attention mechanisms.

Key components include:

- self-attention  
- multi-head attention  
- positional encoding  
- feed-forward networks  
- encoder-based models for classification  

---

### Pre-trained Language Models

Introduction to models trained on large corpora and adapted to downstream tasks.

Topics include:

- fine-tuning vs feature extraction  
- transformer-based encoders (e.g., BERT-like models)  
- domain adaptation  

---

### Efficient Fine-Tuning (LoRA)

Modern techniques to adapt large models efficiently.

Key concepts include:

- parameter-efficient fine-tuning  
- low-rank adaptation (LoRA)  
- reducing computational cost  

---

### Deployment of NLP Models

Basic concepts for taking models into production:

- inference pipelines  
- model serialization  
- simple deployment workflows  

---

# Notebook Exercise – Tweet Classification Pipeline

The practical exercise for this lecture is a **complete NLP pipeline** using the **TweetEval dataset**, where students progressively build and improve a text classification system.

### Exercise Notebooks

https://github.com/jdmartinev/SI7011-DeepLearning/tree/main/Lecture06/notebooks/excercise

The exercise is structured as a sequence of notebooks:

1. **Data Exploration**
   - `tweeteval-part-1-data.ipynb`  
   Understanding the dataset, labels, and basic preprocessing.

2. **Pipeline Construction**
   - `tweeteval-part-2-pipeline.ipynb`  
   Building the data pipeline: tokenization, batching, and preprocessing.

3. **Baseline Transformer Model**
   - `tweeteval-part-3-distilbert.ipynb`  
   Fine-tuning a lightweight transformer (DistilBERT).

4. **Improved Model**
   - `tweeteval-part-4-bertweet.ipynb`  
   Using a domain-specific model for tweets.

5. **Efficient Fine-Tuning**
   - `tweeteval-part-5-lora.ipynb`  
   Applying LoRA for parameter-efficient adaptation.

6. **Deployment**
   - `tweeteval-part-6-deployment.ipynb`  
   Building a simple inference/deployment pipeline.

---

## Exercise Objective

In this exercise students will build a **complete deep learning pipeline for text classification**.

Students will:

- preprocess and analyze text data  
- build NLP pipelines for model training  
- fine-tune **pre-trained transformer models**  
- compare general vs domain-specific models  
- apply **efficient fine-tuning (LoRA)**  
- deploy a trained model for inference  

The exercise emphasizes **end-to-end system design**, not only model training.

---

## Deliverable

Students must submit a **GitHub repository** containing the completed notebooks.

### Requirements

The repository should include:

- All **executed notebooks** (with outputs visible)  
- Completion of the full pipeline:
  - data exploration  
  - preprocessing pipeline  
  - at least one transformer model  
- Experiments showing:
  - comparison between DistilBERT and BERTweet  
  - impact of LoRA vs standard fine-tuning  
- A working **inference or deployment example**  
- Proper organization and readability  

---

### Evaluation Criteria

Submissions will be evaluated based on:

- **Correct implementation of the full pipeline**  
- **Use of transformer-based models**  
- **Understanding of model improvements (DistilBERT vs BERTweet vs LoRA)**  
- **Quality of analysis and comparisons**  
- **Code quality and organization**  
- **Reproducibility**  

Students are encouraged to document insights and decisions directly in the notebooks.
