
# AB6 – Towards Smarter Agriculture: Deep Learning-Based Multistage Detection of Leaf Diseases

## Team Info
- 22471A0533 — **KOYYALAMUDI VENKATA RAMESH** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Dataset preprocessing, EfficientNet-B0 model training, performance evaluation, and project documentation.

- 22471A05XX — **BANGARU SURYA PRASAD** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Feature extraction using PCFAN, disease segmentation with Red Fox Optimization, and result analysis.

- 22471A05XX — **THULLIBILLI NAGAIAH** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Data augmentation, preprocessing pipeline development, and model testing.

- 22471A05XX — **SHAIK ABDUL NABI** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Literature survey, performance metrics analysis, and paper formatting.

---

## Abstract
Plant diseases significantly affect agricultural productivity and farmer income. This project proposes a deep learning-based multistage framework for early and accurate detection of plant leaf diseases using image data. The PlantVillage dataset was utilized, with 10 balanced disease classes selected for effective training. Image preprocessing techniques such as resizing, normalization, grayscale conversion, denoising, and CLAHE were applied to enhance disease features. EfficientNet-B0 with transfer learning was used for classification, achieving 97.9% testing accuracy and 96.2% validation accuracy. Diseased regions were segmented using Red Fox Optimization, obtaining segmentation accuracy above 94%. The proposed system enables reliable and resource-efficient plant disease detection for smart agriculture applications.

---

## Paper Reference (Inspiration)
👉 **[Paper Title Real-Time Plant Disease Dataset Development and Detection of Plant Disease Using Deep Learning
  – Author Names :Diana Susan Joseph; Pranav M. Pawar; Kaustubh Chakradeo
 ](https://ieeexplore.ieee.org/document/10414062)**
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
Balanced class selection (10 classes with equal samples) to avoid bias
Combined classification + segmentation in a unified pipeline
Used EfficientNet-B0 for high accuracy with low computational cost
Applied Red Fox Optimization for precise disease region segmentation
Achieved consistently high Precision, Recall, F1-Score (>95%)

---

## About the Project
What the project does:
Detects and classifies plant leaf diseases from images and highlights affected regions.

Why it is useful:
Helps farmers identify diseases early, reduce crop loss, and improve yield with minimal resources.

Project Workflow:
Leaf Image → Preprocessing → Feature Extraction → EfficientNet-B0 Model → Disease Classification + Segmentation → Output Result

---

## Dataset Used
👉 **[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data  

)**

**Dataset Details:**
~89,000 leaf images
38 plant disease categories
10 classes selected (200 images each) for balanced learning
Includes healthy and diseased leaves (tomato, grape, corn, cherry, squash, etc.)
---

## Dependencies Used
Python, TensorFlow, Keras, OpenCV (cv2), NumPy, Matplotlib, Scikit-learn, Google Colab
---

## EDA & Preprocessing
Image resizing to 224×224
Normalization of pixel values (0–1)
Grayscale conversion
Denoising to remove blur and grain
Duplicate image removal
Data augmentation (rotation, flipping, random cropping)
---

## Model Training Info
Model: EfficientNet-B0 (Transfer Learning)
Optimizer: Adam
Loss Function: Categorical Cross-Entropy
Training Platform: Google Colab (GPU enabled)
Dataset split: Train / Validation / Test (balanced)

---

## Model Testing / Evaluation
Accuracy (Test): 97.9%
Accuracy (Validation): 96.2%
Precision, Recall, F1-Score: >95% for most classes
ROC-AUC: 0.95 – 1.00 across classes
Confusion matrix shows strong class separation

---

## Results
Excellent classification of Powdery Mildew, Late Blight, and Healthy leaves
Tomato Target Spot showed slightly lower recall due to symptom overlap
Average Precision: 96%
Average Recall: 94.2%
Average F1-Score: 95%
Average IoU (segmentation): 92.6%
---

## Limitations & Future Work
Performance may vary in real-world outdoor conditions
Similar disease symptoms can cause minor misclassification
Future enhancements:
Real-time mobile application
Field-level testing
Exact infected-spot marking using advanced segmentation
Edge-device deployment
---

## Deployment Info
Model trained and tested in Google Colab
Can be deployed as:
Web application
Mobile app for farmers
Edge-based smart agriculture system

---
