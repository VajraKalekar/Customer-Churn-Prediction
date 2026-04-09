import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model and columns
# @st.cache_resource prevents the model from reloading every time you move a slider
@st.cache_resource
def load_model_data():
    model = joblib.load('churn_model.pkl')
    cols = joblib.load('model_columns.pkl')
    return model, cols

loaded_model, model_cols = load_model_data()

# 2. UI Header Setup
st.title("🏦 Customer Churn Prediction")
st.markdown("A Decision Support System that predicts risk and recommends retention strategies automatically.")
st.markdown("---")

# 3. Create Sliders for Input
total_trans_amt = st.slider("Total Transaction Amount ($)", min_value=0, max_value=20000, value=2000)
total_trans_ct = st.slider("Total Transaction Count", min_value=0, max_value=150, value=40)
total_revolving_bal = st.slider("Total Revolving Balance ($)", min_value=0, max_value=3000, value=0)
contact_count = st.slider("Contacts with Support", min_value=0, max_value=6, value=3, step=1)

st.markdown("---")

# 4. Prediction Logic (Triggered by a button)
if st.button("Assess Customer Risk", type="primary"):
    
# Prepare Data
    input_data = pd.DataFrame(index=[0], columns=model_cols)
    input_data = input_data.fillna(0)
    input_data['Total_Trans_Amt'] = total_trans_amt
    input_data['Total_Trans_Ct'] = total_trans_ct
    input_data['Total_Revolving_Bal'] = total_revolving_bal
    input_data['Contacts_Count_12_mon'] = contact_count

    # Get Prediction
    churn_prob = loaded_model.predict_proba(input_data)[1]
    churn_percentage = round(churn_prob * 100, 1)

    # 5. Dynamic Strategy Output
    st.subheader("AI Recommendation")
    
    if churn_percentage < 30:
        st.success(f"🟢 **Low Risk ({churn_percentage}%)**")
        st.markdown("""
        **✅ STRATEGY: UPSELL**
        * This customer is happy. Don't offer discounts (saves money).
        * **Action:** Offer Platinum Card upgrade.
        """)
        
    elif 30 <= churn_percentage < 70:
        st.warning(f"🟡 **Medium Risk ({churn_percentage}%)**")
        st.markdown("""
        **💡 STRATEGY: NURTURE**
        * This customer is wobbling. Show them value.
        * **Action:** Send 'Year in Review' email & offer a bundle deal.
        """)
        
    else:
        st.error(f"🔴 **High Risk ({churn_percentage}%)**")
        st.markdown("""
        **⚠️ STRATEGY: RESCUE**
        * Danger Zone! They are about to leave.
        * **Action:** Auto-trigger 15% discount email NOW.
        """)
