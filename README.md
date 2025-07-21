# 📚 Multi-Label Predictions on Academic Articles

This project focuses on automatically classifying academic research papers into multiple relevant subjects using **Natural Language Processing (NLP)** and **Multi-Label Classification** techniques.

---

## 🔍 Project Objective

To build a system that can:
- Analyze the abstract and title of a research paper
- Assign multiple subject labels such as **Computer Science**, **Mathematics**, **Physics**, etc.
- Improve academic content discovery through efficient categorization

---

## 🧾 Dataset Overview

- **Source**: Academic article abstracts and titles
- **Labels**: Each article may belong to one or more of 6 subjects:
  - Computer Science
  - Mathematics
  - Physics
  - Statistics
  - Quantitative Biology
  - Quantitative Finance
- **Size**:
  - 20,972 training samples
  - 8,989 testing samples
  - No missing labels — every article has at least one label
  - 26,267 total label assignments (due to multi-label nature)

---

## 🛠️ Preprocessing Steps

- Merged `Title` and `Abstract` into one `Text` column
- Cleaned text: removed HTML tags, punctuation, special characters
- Applied **stemming** and **stopword removal**
- Extracted additional features:
  - Text length
  - Word count
  - Avg word length
  - Numerals in title

- Addressed class imbalance using **Dynamic MLSMOTE** to generate synthetic samples for underrepresented labels

---

## 🧠 Approaches Used

### ✅ Machine Learning (ML)
- **Vectorization**: TF-IDF, Word2Vec
- **Algorithms**: Logistic Regression, Random Forest, SVM
- **Transformation Methods**: Binary Relevance, Label Powerset, Classifier Chains

### ✅ BERT (Transformer-based Model)
- Used `bert-base-uncased` from HuggingFace
- Custom PyTorch model with sigmoid + BCEWithLogitsLoss for multi-label output
- Achieved **best overall performance**

---

## 🧪 Key Results

- **TF-IDF** outperformed Word2Vec in most ML models
- **BERT** gave the best accuracy and generalization across all tested methods
- Balanced data consistently improved performance across models

---

## ⚠️ Challenges Faced

- Severe **class imbalance** across certain topics
- Limited **GPU resources** restricted experimentation with larger LLMs like RoBERTa or GPT
- Extended runtimes for some transformation-based models

---

## 🔭 Future Work

- Explore **hybrid models** combining ML and DL
- Use **advanced transformers** (e.g., RoBERTa, XLNet)
- Develop smarter **imbalance-handling techniques**
- Implement **stacking and boosting** ensemble methods
