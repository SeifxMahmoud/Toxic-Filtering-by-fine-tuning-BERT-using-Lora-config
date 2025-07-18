# Toxic-Filtering-by-fine-tuning-BERT-using-Lora-config

# 🧹 Text Classification Pipeline with LSTM & DistilBERT

This project presents a complete NLP pipeline for text classification, starting from raw data ingestion and exploration, through preprocessing and augmentation, to model building and deployment. The pipeline addresses common challenges in real-world datasets including class imbalance, noisy text, and overfitting. Two models are explored: an LSTM-based baseline and a fine-tuned DistilBERT using LoRA.

---

## 🗂️ Project Highlights

- ✅ **Exploratory Data Analysis (EDA)**
- ✅ **Text Cleaning & Preprocessing**
- ✅ **Data Augmentation with EDA Techniques**
- ✅ **Label Encoding & Tensor Transformation**
- ✅ **Train/Val/Test Splits and Sequence Length Analysis**
- ✅ **LSTM Model with Regularization and Callbacks**
- ✅ **DistilBERT Fine-tuning using LoRA**
- ✅ **Evaluation with Classification Metrics**
- ✅ **Deployment via Streamlit (LLaMA Guard + BLIP)**

---

## 🧪 Dataset Insights

During EDA, the following observations and actions were taken:

- Significant **duplicate rows** were found — these were dropped to mitigate data leakage.
- **Sparse categories** with little to no samples were merged into a single `"other"` category.
- The target column was renamed from `category` to `label` for clarity and downstream compatibility.
- **Class imbalance** was noted and addressed using **Easy Data Augmentation (EDA)** techniques.

---

## 🔧 Preprocessing Steps

1. **Text Consolidation**: Combined multiple columns into a unified `'text'` column.
2. **Text Cleaning**: Used regular expressions to:
   - Strip whitespace
   - Lowercase text
   - Remove accented characters
3. **Stopword Removal & Stemming**:
   - Applied `nltk.PorterStemmer`
   - Removed standard stopwords
4. **Label Encoding**: Target labels were numerically encoded and added to the DataFrame.
5. **Train/Val/Test Split**:
   - Stratified sampling
   - Converted splits to tensors
6. **Sequence Length Estimation**:
   - Percentile analysis was used to set a cutoff for sentence lengths.
   - Informed `TextVectorization` configuration.

---

## 🧠 Models

### 🔹 LSTM Baseline

- Built using `tf.keras.Sequential`
- Inputs were pre-tokenized and vectorized
- Used callbacks (e.g., EarlyStopping, ModelCheckpoint)
- Regularized with:
  - **Dropout layers**
  - **Ridge (L2) regularization** on dense layers
  - **Neuron reduction** to prevent overfitting

### 🔹 DistilBERT (Fine-tuned)

- Used Hugging Face Transformers with **LoRA configuration**
- Tokenized inputs using `AutoTokenizer` (with padding, truncation)
- Fine-tuned on augmented dataset

---

## 📊 Evaluation

- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix**
- Performance was tracked across both models
- Overfitting in LSTM was resolved with architectural and regularization adjustments

---

## 🖥️ Deployment

A `Streamlit` dashboard was created to demonstrate:

- **LSTM predictions**
- **DistilBERT predictions**
- **Integration with LLaMA Guard** for safety filtering
- **BLIP (Bootstrapped Language-Image Pretraining)** for contextual visualization

---

## 🧪 Key Learnings

- Careful **EDA** can uncover serious issues like leakage from duplicates and class imbalance.
- Basic NLP cleaning still plays a role before BERT-class models, especially for exploratory baselines.
- LSTM models are prone to overfitting without careful architecture design and regularization.
- LoRA allows efficient fine-tuning of large models like DistilBERT with minimal compute.
- Tokenization strategy and sequence length estimation impact both model performance and compute cost.
- Data augmentation like EDA can significantly improve class balance and generalization.

---

## 📁 Technologies Used

| Component        | Library/Tool          |
|------------------|------------------------|
| Data Handling    | `pandas`, `nltk`       |
| Modeling         | `TensorFlow`, `transformers` |
| Augmentation     | EDA (Easy Data Augmentation) |
| Evaluation       | `sklearn.metrics`      |
| Deployment       | `Streamlit`            |
| Additional Models| LLaMA Guard, BLIP      |

---

## 🚀 How to Run

### Clone and set up environment

```bash
git clone https://github.com/yourusername/text-classification-lstm-distilbert.git
cd text-classification-lstm-distilbert
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
