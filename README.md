# **Machine Learning : Sentiment Analysis with Naive Bayes & Active Learning**  

## **Overview**  
This repository contains the implementation of a **Multinomial Naive Bayes (MNB) classifier** for sentiment analysis on the **Sentiment140 dataset**, along with an **Active Learning Strategy (ALS)** to optimize data labeling efficiency.  

### **Key Features**  
- **Multinomial Naive Bayes (MNB)** for binary sentiment classification (positive/negative).  
- **Active Learning Strategy (ALS)** using **uncertainty sampling** to reduce labeling effort.  
- Comparison with **random selection baseline**.  
- **Incremental model updates** for efficient retraining.  

---

## **1. Multinomial Naive Bayes (MNB) Classifier**  

### **Algorithm**  
1. **Count Vectorization**  
   - Convert text into a sparse matrix of word counts (top 10,000 frequent words).  
   -  **TF-IDF** feature representation:  
     \[
     \text{TF-IDF}(w,d) = \text{TF}(w,d) \times \text{IDF}(w)
     \]
     where:  
     - \(\text{TF}(w,d)\) = Term frequency in document \(d\).  
     - \(\text{IDF}(w)\) = Inverse document frequency.  

2. **Training**  
   - Compute **class priors** \(P(c)\):  
     \[
     P(c) = \frac{\text{Number of documents in class } c}{\text{Total documents}}
     \]
   - Compute **smoothed class-conditional probabilities** \(P(w \mid c)\):  
     \[
     P(w \mid c) = \frac{\text{count}(w,c) + \alpha}{\text{total count}(c) + \alpha \times \text{vocab size}}
     \]  
     (Laplace smoothing with \(\alpha = 0.1\))  

3. **Inference**  
   - Predict class using **log posterior probabilities**:  
     \[
     \log P(c \mid d) = \log P(c) + \sum_{w \in d} \text{count}(w,d) \cdot \log P(w \mid c)
     \]  

### **Assumptions**  
- **Conditional independence** of words given the class.  
- **Bag-of-words** representation (word order ignored).  

### **Implementation**  
- **`Vectorizer` class**: Handles vocabulary extraction and text vectorization.  
- **`MultinomialNaiveBayes` class**: Implements training, inference, and incremental updates.  
- **`prob1.py`**: Preprocesses data, trains MNB, and evaluates on validation/test sets.  

### **Results**  
| Metric          | Accuracy (%) |  
|-----------------|-------------:|  
| Training        | 85.2         |  
| Validation      | 82.7         |  
| Test (Leaderboard) | 81.5         |  

---

## **2. Active Learning Strategy (ALS)**  

### **Algorithm**  
1. **Uncertainty Sampling**  
   - Selects data points with highest **entropy**:  
     \[
     \text{Entropy}(d) = -\sum_c P(c \mid d) \log P(c \mid d)
     \]  
2. **Incremental Training**  
   - Updates model with newly labeled data without full retraining.  
3. **Comparison with Random Selection**  
   - Evaluates efficiency gains of ALS over random labeling.  

### **Implementation**  
- **`prob2.py`**: Runs active learning experiments.  
- **Uncertainty calculation** via `predict_proba`.  
- **Model updates** using modified `fit` method in `MultinomialNaiveBayes`.  

### **Results**  
- ALS achieved **baseline accuracy (82.7%) with 50% less data** than random selection.  
- **Faster convergence** in early training stages.  
- **Computationally efficient** (avoids retraining from scratch).  

---

## **Assumptions**  
- Dataset is **balanced** (no class prior adjustments).  
- Vocabulary limited to **top 10,000 words** for memory efficiency.  
- **Laplace smoothing** (\(\alpha = 0.1\)) prevents zero probabilities.  
- Newly labeled data is **correct**.  

---

## **Usage**  
1. **Naive Bayes Training & Evaluation**:  
   ```bash
   python prob1.py
   ```  
2. **Active Learning Experiment**:  
   ```bash
   python prob2.py
   ```  

---
