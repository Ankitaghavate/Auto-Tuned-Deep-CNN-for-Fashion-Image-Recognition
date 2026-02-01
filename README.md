# Auto-Tuned Deep CNN for Fashion Image Recognition

## 📌 Project Overview
This project implements a **high-accuracy fashion image classification system** using  
**transfer learning with a pretrained VGG16 model** in **PyTorch**.

The goal is to improve classification performance on the **Fashion-MNIST dataset** by:
- Leveraging pretrained ImageNet features
- Applying correct preprocessing and normalization
- Fine-tuning a custom classifier head
- Visualizing training performance with clear charts

The final model achieves **above 90% accuracy** on the test dataset.

---

## 🎯 Objectives
- Use **transfer learning** to boost CNN performance
- Convert grayscale Fashion-MNIST images to **RGB** for compatibility with VGG16
- Apply **ImageNet normalization**
- Train and evaluate a deep CNN on GPU
- Visualize accuracy and loss trends

---

## 🧠 Model Architecture
- **Backbone:** Pretrained VGG16 (ImageNet)
- **Frozen layers:** Convolutional feature extractor
- **Trainable layers:** Custom fully connected classifier
- **Output:** 10 fashion categories

---

## 🛠️ Technologies Used
- Python  
- PyTorch  
- Torchvision  
- Scikit-learn  
- Pandas  
- Matplotlib  
- Google Colab / GPU  

---

## 📂 Dataset
- **Fashion-MNIST**
  - 28×28 grayscale images
  - 10 clothing classes
- Images are converted to **3-channel RGB** before training

---

## ⚙️ Workflow
1. Load and preprocess Fashion-MNIST data  
2. Convert grayscale images to RGB  
3. Resize and normalize images using ImageNet statistics  
4. Load pretrained VGG16 model  
5. Fine-tune the classifier layers  
6. Evaluate model performance  
7. Visualize results using charts  

---

## 📊 Results & Visualizations
The notebook includes:
- Training vs Validation Accuracy curve
- Training Loss vs Epochs curve
- Final Train vs Test Accuracy comparison bar chart

These plots clearly show model convergence and strong generalization.

---

## 🚀 How to Run
1. Open the notebook:
2. Upload `fmnist_small.csv` to your runtime
3. Run all cells (GPU recommended)

---

## 📈 Final Performance
- **Test Accuracy:** > 90%
- Stable training and low overfitting due to transfer learning

---

## 🔮 Future Improvements
- Replace VGG16 with ResNet18 or EfficientNet
- Add Optuna for automated hyperparameter tuning
- Include confusion matrix and class-wise accuracy

---
