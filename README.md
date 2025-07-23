
# Student Performance Prediction 🎓

**Predicting student outcomes using real-world data & machine learning**

## 🔍 Project Overview

This project explores how well we can predict:
- Whether a student will **pass or fail** (classification task)
- Their **final grade (G3)** on a 0–20 scale (regression task)

Using data-driven analysis, feature engineering, model training, and real-world testing, this project demonstrates how to build an end-to-end student performance predictor.

---

## 📁 Repository Structure

- **`Student_Performance_Prediction.ipynb`** – Your complete analysis: EDA, feature engineering, modeling, visualizations, and conclusions  
- **`predict_student.py`** – A script to input a new student's data and get pass/fail and grade predictions  
- **`/models`** – Saved files: trained SVM, linear regression model, and scalers used in preprocessing  
- **`requirements.txt`** – All dependencies needed to run the project  
- **`README.md`** – This introductory file  

---

## 📊 Key Results

| Task         | Model                 | Test Performance                      |
|--------------|-----------------------|---------------------------------------|
| Classification | **SVM**             | 91% accuracy (best performance)        |
| Regression     | **Linear Regression** | MAE ≈ 1.45, R² ≈ 0.76 (strong predictor) |

**Top Influencing Features:**  
- `G2`, `G1` (recent grades), `parent_edu`, `traveltime`, and `failures`

---

## 🧩 Install & Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Jeylani-2526/student-performance-prediction.git
   cd student-performance-prediction


2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. To run the prediction script:

   ```bash
   python predict_student.py
   ```

   Or launch the notebook:

   ```bash
   jupyter lab
   ```

4. Open and explore the notebook:

   * Step through EDA and feature engineering
   * Run modeling cells
   * Check visualization and conclusions

---

## 🧠 What I Learned

* **Academic history matters**: Prior grades are the strongest predictors of final performance
* **Balanced modeling approach**: Combining simplicity (logistic, linear) with complexity (SVM, Random Forest) improved both understanding and accuracy
* **Interpretability matters**: Feature importance and visualizations translate models into actionable insights for educators

---

## 🚀 Next Steps

* Add additional behavioral or attendance data to improve model robustness
* Deploy a lightweight web app for interactive prediction
* Share insights with schools or educational platforms to support data-driven intervention

---

## 🙋 About Me

I’m **\[Your Name]**, aspiring Junior Data Scientist with a passion for using data to improve educational outcomes. Feel free to connect on [LinkedIn](https://www.linkedin.com/in/abdullahi-jeylani-1a7b83278/) or email me at [abdallamuhammed07@gmai.com](abdallamuhammed07@gmail.com).

---

Thank you for exploring the project! Feel free to ⭐ the repo if you find it useful.

```


