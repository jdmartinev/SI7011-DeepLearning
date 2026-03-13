# Deep Learning – Classes 3 & 4  
## Convolutional Neural Networks, Data Augmentation, Transfer Learning, and Self-Supervised Learning

This repository contains the material for **Classes 3 and 4 of the Deep Learning course**.

In these sessions we move from fully connected neural networks to **deep learning models designed for image data**. The lectures introduce **Convolutional Neural Networks (CNNs)** and discuss how they exploit spatial structure to learn hierarchical visual representations.

We also cover **data augmentation**, which improves model generalization, and **transfer learning**, where models pre-trained on large datasets are adapted to new tasks.

Finally, we introduce the concept of **self-supervised learning**, focusing on **SimCLR**, a modern approach for learning visual representations without labeled data.

The lectures are accompanied by slides, example code, and a practical **image classification competition using the Tiny ImageNet dataset**.

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

### Data Augmentation

Techniques used to artificially expand the training dataset and improve model robustness.

Common augmentation strategies include:

- random flips  
- rotations  
- random crops  
- color jitter  
- normalization  

These techniques help reduce overfitting and improve generalization.

---

### Transfer Learning

Using models **pre-trained on large image datasets** and adapting them to new tasks.

Topics include:

- feature extraction with frozen networks  
- fine-tuning pre-trained CNNs  
- replacing classification heads  
- practical strategies for small datasets  

Transfer learning is one of the most widely used approaches in applied deep learning.

---

### Self-Supervised Learning

Introduction to methods that learn representations **without requiring labeled data**.

We discuss the motivation behind self-supervised learning and its role in modern representation learning.

---

### SimCLR

SimCLR is a **contrastive self-supervised learning method** that learns image representations by maximizing agreement between augmented views of the same image.

Key ideas include:

- contrastive learning  
- positive and negative pairs  
- projection heads  
- representation learning without labels  

---

# Notebook Exercise – Tiny ImageNet Competition

The practical exercise for these lectures is a **multi-class image classification task** using the **Tiny ImageNet dataset**.

Tiny ImageNet is a reduced version of ImageNet containing **200 object categories and 100,000 images resized to 64×64 pixels**. Each class includes **500 training images, 50 validation images, and 50 test images**, making it a common benchmark for deep learning experiments. :contentReference[oaicite:1]{index=1}  

Students will train and evaluate convolutional neural networks on this dataset.

### Exercise Notebook

https://www.kaggle.com/code/juanmartinezv4399/si7011-tinyimagenetcompetition

---

## Exercise Objective

In this exercise students will build and train **deep learning models for image classification using PyTorch**.

Students will explore:

- training **Convolutional Neural Networks** for image classification  
- implementing **data augmentation pipelines**  
- improving performance using **regularization techniques**  
- applying **transfer learning with pre-trained CNNs**  
- experimenting with strategies to improve classification accuracy  

The competition format encourages students to **iteratively improve their models**, test different architectures, and explore techniques commonly used in modern computer vision pipelines.
