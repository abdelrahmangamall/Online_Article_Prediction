# Machine Learning Project: Regression and Classification

This project involves two phases: **Regression** and **Classification**, where various machine learning techniques are applied to preprocess data, select features, and build models. Below is an overview of the project structure, techniques used, and key insights.

---

## **Phase 1: Regression**

### **Preprocessing Techniques**
- **Handling Missing Values:** Null values in the dataset were checked and dropped.
- **One-Hot Encoding:** Applied to categorical features (`channel type`, `weekday`, `isWeekEnd`).
- **Feature Scaling:** Standardized features using `StandardScaler`.
- **Feature Selection:** Used `SelectKBest` with the `r_regression` score function to select the top 10 features based on their relevance to the target variable.

### **Selected Features**
The top 10 features selected using `SelectKBest` are:
- `num_hrefs`
- `kw_avg_max`
- `kw_min_avg`
- `kw_max_avg`
- `kw_avg_avg`
- `self_reference_min_shares`
- `self_reference_max_shares`
- `self_reference_avg_shares`
- `LDA_02`
- `LDA_03`

### **Regression Techniques**
1. **Linear Regression:**
   - Test MSE: **162,793,117.45**
   - Test R-squared: **0.0167**

2. **Random Forest Regression:**
   - Test MSE: **168,930,749.93**
   - Test R-squared: **-0.0204**

### **Results**
- **Training Set Size:** 80% of the dataset
- **Testing Set Size:** 20% of the dataset
- Both regression models were implemented and evaluated using MSE and R-squared metrics. Further improvements can be explored by fine-tuning hyperparameters and incorporating advanced feature engineering techniques.

---

## **Phase 2: Classification**

### **Preprocessing Techniques**
- **Handling Missing Values:** Null values in the dataset were checked and dropped.
- **Label Encoding:** Applied to categorical features (`channel type`, `weekday`, `isWeekEnd`).
- **Feature Scaling:** Standardized features using `StandardScaler`.

### **Classification Techniques**
1. **Random Forest Classification:**
   - Testing Time: **0.588 seconds**
   - Hyperparameters:
     - `N Estimators`: 200
     - `Random State`: 42

2. **AdaBoost Classification:**
   - Testing Time: **0.183 seconds**
   - Hyperparameters:
     - `N Estimators`: 100
     - `Algorithm`: `SAMME`

3. **SVM Classification:**
   - Testing Time: **12.464 seconds**
   - Hyperparameters:
     - `C`: 1
     - `gamma`: 0.1
     - `Kernel`: `linear`

### **Training and Testing Times**
- **Random Forest:**
  - Training Time: **52.838 seconds**
  - Testing Time: **0.588 seconds**
- **AdaBoost:**
  - Training Time: **16.768 seconds**
  - Testing Time: **0.183 seconds**
- **SVM:**
  - Training Time: **564.288 seconds**
  - Testing Time: **12.464 seconds**

---

## **Conclusion**

### **Phase 1: Regression**
- Preprocessing included handling missing values, one-hot encoding, feature scaling, and feature selection.
- Two regression models (Linear Regression and Random Forest Regression) were implemented and evaluated using MSE and R-squared metrics.
- Further improvements can be made by exploring additional regression techniques and fine-tuning hyperparameters.

### **Phase 2: Classification**
- Preprocessing included handling missing values, label encoding, and feature scaling.
- Three classification models (Random Forest, AdaBoost, and SVM) were trained and evaluated based on training and testing times.
- The project demonstrates the importance of preprocessing and feature engineering in building machine learning models.

### **Next Steps**
- Explore additional regression and classification techniques.
- Fine-tune hyperparameters for better model performance.
- Incorporate advanced feature engineering methods to improve results.

---

## **Project Structure**
- **Data Preprocessing:** Handling missing values, encoding categorical features, and scaling.
- **Feature Selection:** Using `SelectKBest` to select the most relevant features.
- **Model Training:** Implementing Linear Regression, Random Forest Regression, Random Forest Classification, AdaBoost, and SVM.
- **Model Evaluation:** Evaluating models using training and testing times.
- **Saving Models:** Trained models are saved as pickle files for future use.

---

## **How to Use**
1. **Preprocess Data:** Use the provided preprocessing steps to clean and prepare the dataset.
2. **Train Models:** Run the regression and classification scripts to train the models.
3. **Evaluate Models:** Evaluate the models using the provided metrics (e.g., training and testing times).
4. **Save and Load Models:** Save trained models as pickle files and load them for predictions on new data.

---

## **Dependencies**
- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---



###Report Link: https://drive.google.com/file/d/1eWVoQ-NQhUEABdjk9yPdW_GlxaH2b26k/view?usp=drive_link
