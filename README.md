🏦 Customer Churn Prediction
A machine learning web application that predicts the likelihood of a bank customer churning and automatically recommends a business retention strategy.
🚀 Live Demo → Click Here (replace with your actual Streamlit URL](https://customer-churn-predictive-model.streamlit.app/)

📌 What This Project Does
Banks lose millions every year to customer churn. This app gives a relationship manager or analyst a real-time risk score for any customer — and then tells them exactly what to do about it.
Enter 4 key customer signals, hit Assess Customer Risk, and the model instantly outputs:

A churn probability score
A color-coded risk tier (Low / Medium / High)
A specific business action to take (Upsell / Nurture / Rescue)


🖼️ App Preview

<img width="1110" height="782" alt="Screenshot 2026-04-10 184036" src="https://github.com/user-attachments/assets/7f47e193-c176-4c66-82fc-e0278fe00df6" />
<img width="1252" height="512" alt="Screenshot 2026-04-10 184055" src="https://github.com/user-attachments/assets/95773722-8649-4eb0-912b-897e6b631a8d" />



🧠 How It Works
Dataset: BankChurners.csv — 10,127 real bank customers with 20+ features including transaction history, credit limits, and account activity.
Pipeline:
Raw Data → Drop Junk Columns → Encode Categoricals → Train/Test Split
→ SMOTE (handle class imbalance) → Random Forest (100 trees) → Saved Model
Key decisions:

Dropped CLIENTNUM and two Naive Bayes pre-scored columns that would leak the answer
Used SMOTE to fix the 84/16 class imbalance (most customers don't churn — the model would cheat without this)
Random Forest chosen for robustness and built-in feature importance
Model achieves ~96% accuracy on the held-out test set


🔍 Top Churn Predictors (Feature Importance)
The model identified these as the strongest signals:
FeatureWhy It MattersTotal_Trans_CtLow transaction count = disengaging customerTotal_Trans_AmtDropping spend = customer pulling awayTotal_Revolving_BalLow balance = not using the credit productContacts_Count_12_monHigh support contact = frustrated customer

🗂️ Repository Structure
Customer-Churn-Prediction/
│
├── app.py                          # Streamlit web app
├── customer_churn_prediction_project.py  # Full ML pipeline (training code)
├── BankChurners.csv                # Raw dataset
├── churn_model.pkl                 # Trained Random Forest model
├── model_columns.pkl               # Feature column names (for inference)
├── requirements.txt                # Python dependencies
└── README.md

⚙️ Run Locally
bash# 1. Clone the repo
git clone https://github.com/VajraKalekar/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
Open http://localhost:8501 in your browser.

📦 Tech Stack
LayerToolLanguagePython 3MLscikit-learn, imbalanced-learnDatapandas, numpyVisualizationmatplotlib, seabornAppStreamlitModel PersistencejoblibDeploymentStreamlit Community Cloud

📊 Model Performance
MetricScoreAccuracy~96%Precision (Churn)HighRecall (Churn)High (boosted by SMOTE)AlgorithmRandom Forest (100 estimators)

💡 Business Strategy Logic
Risk ScoreLabelRecommended Action< 30%🟢 Low RiskUpsell — Offer Platinum Card upgrade30–70%🟡 Medium RiskNurture — Send Year in Review email + bundle deal> 70%🔴 High RiskRescue — Auto-trigger 15% discount immediately

👤 Author
Vajra Kalekar(https://github.com/VajraKalekar)
GitHub

📄 License
This project is open source and available under the MIT License.
