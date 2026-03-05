# Deep Learning – Class 2  
## Multi-Layer Perceptrons, Cross Entropy, and Regularization with PyTorch

This repository contains the material for **Class 2 of the Deep Learning course**.

In this session we extend the ideas introduced in the first class and move from single-neuron models to **multi-layer neural networks**. The class introduces **Multi-Layer Perceptrons (MLPs)**, discusses the **cross entropy loss used in multi-class problems**, and presents basic **regularization techniques** used to improve generalization.

The lecture is accompanied by slides, example code, and a practical exercise implemented in a Kaggle notebook.

---

## Topics Covered

This class focuses on the following concepts:

- **Multi-Layer Perceptrons (MLPs)**  
  Neural networks with one or more hidden layers that allow the model to learn non-linear decision boundaries.

- **Activation Functions**  
  Use of non-linearities (e.g., ReLU) to increase model expressiveness.

- **Cross Entropy Loss**  
  Extension from binary cross entropy to **multi-class classification** using softmax and cross entropy.

- **Training Neural Networks**  
  Forward pass, loss computation, backpropagation, and parameter updates.

- **Regularization Techniques**  
  Methods to reduce overfitting, including:
  - weight decay (L2 regularization)
  - dropout
  - general strategies for improving model generalization.

- **Building MLPs with PyTorch**  
  Implementing multi-layer networks using `torch.nn`.

---

## Class Materials

This class includes the following resources:

- **Slides** – theoretical explanation of multi-layer neural networks and training.
- **Example Code** – demonstrations of MLP architectures and training behavior.
- **Notebook Exercise** – hands-on implementation of an MLP using PyTorch.

Notebook exercise:

https://www.kaggle.com/code/juanmartinezv4399/si7011-dl-mlp-pytorch

---

## Exercise Objective

The notebook allows students to build and train a **Multi-Layer Perceptron using PyTorch**.

Students will explore:

- defining neural network architectures with hidden layers  
- applying activation functions  
- using cross entropy loss for classification  
- training models with PyTorch optimizers  
- applying basic regularization techniques  

This exercise builds on the previous class and prepares students for **more advanced neural network architectures introduced later in the course**.
