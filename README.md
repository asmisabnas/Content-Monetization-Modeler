# 📊 Content Monetization Modeler

## Overview
This project predicts **YouTube Ad Revenue** for individual videos using regression models. It leverages performance metrics (views, likes, comments, watch time, etc.) and contextual features (category, device, country) to estimate potential ad revenue. The final model is deployed in a **Streamlit web app** for interactive predictions.

---

## ✨ Features
- Data Cleaning (missing values, duplicates)
- Feature Engineering (engagement rate, watch-to-length ratio)
- Exploratory Data Analysis (EDA) with visualizations
- Regression Models: Linear, Ridge, Lasso, Random Forest, Gradient Boosting
- Model Evaluation (R², RMSE, MAE)
- Streamlit App for predictions and analytics

---

## 📂 Dataset
- **Name:** YouTube Monetization Modeler (synthetic dataset)
- **Format:** CSV (~122,000 rows)
- **Target Variable:** `ad_revenue_usd`
- **Key Features:** views, likes, comments, watch_time_minutes, video_length_minutes, subscribers, category, device, country

---

## 🚀 Workflow
1. **Dataset Exploration & EDA**
   - Identify trends, correlations, and outliers.
2. **Preprocessing**
   - Handle missing values (~5%), remove duplicates (~2%), encode categorical variables.
3. **Feature Engineering**
   - Engagement Rate = (likes + comments) / views
   - Watch-to-Length Ratio = watch_time_minutes / video_length_minutes
4. **Model Building**
   - Train multiple regression models.
   - Compare performance using R², RMSE, MAE.
5. **Streamlit App**
   - User inputs video metrics.
   - Predicts ad revenue.
   - Displays visual analytics.

---

## 📈 Results
| Model              | R²    | RMSE   | MAE   |
|--------------------|-------|--------|-------|
| Linear Regression  | 0.9526 | 13.48 | 3.12 |
| Ridge Regression   | 0.9526 | 13.48 | 3.12 |
| Lasso Regression   | 0.9526 | 13.47 | 3.12 |
| Random Forest      | 0.9521 | 13.55 | 3.70 |
| Gradient Boosting  | 0.9518 | 13.58 | 4.07 |

✅ Best performing model: **Linear Regression** (saved as `final_model.pkl`)

---

## 🖥️ Streamlit App
### Run the app:
```bash
streamlit run app.py


## ✨ Features
- Data Cleaning (missing values, duplicates)
- Feature Engineering (engagement rate, watch-to-length ratio)
- Exploratory Data Analysis (EDA) with visualizations
- Regression Models: Linear, Ridge, Lasso, Random Forest, Gradient Boosting
- Model Evaluation (R², RMSE, MAE)
- Streamlit App for predictions and analytics

---

## 📂 Dataset
- **Name:** YouTube Monetization Modeler (synthetic dataset)
- **Format:** CSV (~122,000 rows)
- **Target Variable:** `ad_revenue_usd`
- **Key Features:** views, likes, comments, watch_time_minutes, video_length_minutes, subscribers, category, device, country

---

## 🚀 Workflow
1. **Dataset Exploration & EDA**
   - Identify trends, correlations, and outliers.
2. **Preprocessing**
   - Handle missing values (~5%), remove duplicates (~2%), encode categorical variables.
3. **Feature Engineering**
   - Engagement Rate = (likes + comments) / views
   - Watch-to-Length Ratio = watch_time_minutes / video_length_minutes
4. **Model Building**
   - Train multiple regression models.
   - Compare performance using R², RMSE, MAE.
5. **Streamlit App**
   - User inputs video metrics.
   - Predicts ad revenue.
   - Displays visual analytics.

---

## 📈 Results
| Model              | R²    | RMSE   | MAE   |
|--------------------|-------|--------|-------|
| Linear Regression  | 0.9526 | 13.48 | 3.12 |
| Ridge Regression   | 0.9526 | 13.48 | 3.12 |
| Lasso Regression   | 0.9526 | 13.47 | 3.12 |
| Random Forest      | 0.9521 | 13.55 | 3.70 |
| Gradient Boosting  | 0.9518 | 13.58 | 4.07 |

✅ Best performing model: **Linear Regression** (saved as linear_model.pkl`)

---

## 🖥️ Streamlit App
### Run the app:
```bash
streamlit run app.py
Features:
Input fields for video metrics

Predicted ad revenue in USD

File upload option for batch predictions

Visualizations: Revenue vs Views scatter plot, Feature importance chart

👩‍💻 Author
Sabna Asmi S  
LinkedIn: https://www.linkedin.com/in/asmi-sabna/
