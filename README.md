# 💳 Credit Card Fraud Detection — Machine Learning Project

An academic project implementing an end-to-end Machine Learning pipeline for credit card fraud detection **from first principles** using pure **NumPy**, **Pandas**, and **Matplotlib**. 

Developed in strict compliance with subject evaluation criteria requiring algorithms to be implemented **from scratch without black-box ML library packages** (such as Scikit-Learn or XGBoost).

---

## 📋 Evaluation Criteria Mapping

This repository fulfills all 4 project evaluation requirements:

| Component | Weight | Implementation Details in Repo |
| :--- | :---: | :--- |
| **2(a) Understanding the Theory** | 10% | Mathematical formulation of Binary Cross-Entropy (BCE) Loss, Sigmoid hypothesis, chain-rule analytical gradient derivations, and class imbalance mitigation in `notebooks/fraud_detection.ipynb` and `report/report.tex`. |
| **2(b) Implementation from Scratch** | 20% | Full from-scratch algorithms implemented in pure NumPy without Scikit-learn/XGBoost (`LogisticRegressionScratch`, `StandardScalerScratch`, `undersample_dataset`, `train_test_split_scratch`, and scratch evaluation metrics). |
| **2(c) Experimental Results** | 10% | Quantitative empirical evaluation on held-out test split: Loss convergence curve, Confusion Matrix, Precision, Recall, F1-Score, Trapezoidal ROC-AUC, and threshold sensitivity analysis. |
| **2(d) Report Submission (LaTeX)** | 10% | Professional publication-quality LaTeX report (`report/report.tex`) featuring cover page, team details, date of submission, and individual contribution matrix. |

---

## 📂 Project Structure

```bash
Credit-Card-Fraud-Detection-Model/
│
├── data/
│   ├── creditcard.csv             # Kaggle dataset (place downloaded CSV here)
│   └── live_transactions.csv
│
├── notebooks/
│   └── fraud_detection.ipynb      # Complete academic notebook (Theory + Scratch Code + Visualizations)
│
├── report/
│   └── report.tex                 # Academic project report written in LaTeX
│
├── archive/                       # Archived web-app / simulator artifacts
│   ├── app.py
│   ├── transaction_generator.py
│   └── models/
│
├── requirements.txt               # Pure Python dependencies (numpy, pandas, matplotlib)
└── README.md
```

---

## ⚙️ Installation & Running the Notebook

### 1. Set Up Python Environment
```bash
# Clone the repository
git clone https://github.com/kartik7115-edu/Credit-Card-Fraud-Detection-Model.git
cd Credit-Card-Fraud-Detection-Model

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
1. Download the dataset from [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place `creditcard.csv` in the `data/` folder:
   ```
   data/creditcard.csv
   ```
   *(Note: The notebook contains a fallback data generator so it can execute end-to-end even before the full 150MB CSV is downloaded).*

### 3. Launch the Jupyter Notebook
```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

---

## 📄 Compiling the LaTeX Report

The report is written in standard LaTeX (`report/report.tex`) and can be compiled using any LaTeX distribution (TeX Live, MiKTeX, MacTeX) or online via **Overleaf**.

### Local Compilation via `pdflatex`:
```bash
cd report
pdflatex report.tex
pdflatex report.tex
```

---

## 🧠 Key Theoretical Formulations Implemented

1. **Hypothesis & Sigmoid Activation:**
   $$\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

2. **Binary Cross-Entropy Loss:**
   $$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2m}\|\mathbf{w}\|_2^2$$

3. **Analytical Gradients for Parameter Updates:**
   $$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{m} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}) + \frac{\lambda}{m} \mathbf{w}$$
   $$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$

4. **Composite Trapezoidal ROC-AUC Quadrature:**
   $$\text{AUC} = \sum_{k=1}^{K-1} \frac{\text{TPR}_k + \text{TPR}_{k+1}}{2} \cdot (\text{FPR}_{k+1} - \text{FPR}_k)$$

---

## 👥 Authors & Academic Contribution
Refer to `report/report.tex` (Title Page & Section 6) for group details and individual contribution breakdown.
