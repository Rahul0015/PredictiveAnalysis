# Emotion Detection using SVM

This project uses **Support Vector Machines (SVM)** and **TF-IDF vectorization** to classify text into emotional categories like joy, anger, sadness, etc.

---

## 🗂 Dataset

Make sure your CSV file is named `emotions.csv` and contains:

- A column with text (`text`)
- A column with corresponding emotion labels (`label`)

---

## ⚙️ How to Run

1. Open this folder in VS Code.
2. Make sure your Folders structures look like this
   EmotionClassiffier/
│
├── model/
│   ├── svm_model.joblib
│   └── vectorizer.joblib
│
├── api/                     # Streamlit app
│   └── streamlit_app.py     # Moved here
│
├── emotions.csv      # Moved dataset file directly here
│
├── train_model.py           # Keep for retraining
├── requirements.txt
└── .gitignore


4. Set up virtual environment
5. Install dependencies:

```bash
pip install pandas scikit-learn
pip install numpy
pip install matplotlib
pip install seaborn
pip install streamlit
pip install numpy

```
