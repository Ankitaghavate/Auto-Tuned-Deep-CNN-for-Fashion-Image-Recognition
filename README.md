# Auto-Tuned Deep CNN for Fashion Image Recognition

## 📌 Project Overview
This project focuses on building an **auto-tuned deep Convolutional Neural Network (CNN)** for **fashion image classification** using the **Fashion-MNIST dataset**.  
The main goal is to **improve classification accuracy** by combining **PyTorch-based CNN models**, **transfer learning concepts**, and **automated hyperparameter optimization using Optuna**.

Instead of using a fixed CNN architecture, the model structure and training parameters are **dynamically optimized** to find the best-performing configuration.

---

## 🎯 Objectives
- Design a **dynamic CNN architecture** for image recognition  
- Apply **transfer learning concepts** to enhance feature extraction  
- Use **Optuna** to automatically tune hyperparameters  
- Improve overall **model accuracy and generalization**
- Perform efficient training using **GPU (if available)**

---

## 🧠 Key Features
- Dynamic CNN with configurable:
  - Number of convolution layers
  - Number of filters
  - Kernel sizes
  - Fully connected layers
- **Automated hyperparameter tuning** (learning rate, optimizer, batch size, dropout, etc.)
- **Data augmentation** to reduce overfitting
- Support for multiple optimizers (SGD, Adam, RMSprop)
- Evaluation using test accuracy

---

## 🛠️ Technologies Used
- **Python**
- **PyTorch**
- **Optuna**
- **Torchvision**
- **Scikit-learn**
- **Pandas & Matplotlib**
- **Google Colab / GPU support**

---

## 📂 Dataset
- **Fashion-MNIST**
  - 28×28 grayscale images
  - 10 clothing categories (T-shirt, Trouser, Pullover, etc.)

---

## ⚙️ Workflow
1. Load and preprocess Fashion-MNIST data  
2. Apply data augmentation techniques  
3. Build a dynamic CNN model in PyTorch  
4. Use Optuna to search optimal hyperparameters  
5. Train and evaluate the best model configuration  

---

## 📈 Results
- Achieved improved classification accuracy through:
  - Automated hyperparameter tuning
  - Optimized CNN architecture
  - Regularization and data augmentation

---

## 🚀 Future Improvements
- Apply pretrained CNN models (ResNet, VGG) for deeper transfer learning
- Add early stopping and learning rate schedulers
- Perform detailed performance analysis (confusion matrix, F1-score)
