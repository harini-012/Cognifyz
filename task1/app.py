import streamlit as st 
import pandas as pd
import pickle 
model=pickle.load(open("task1/model.pkl","rb"))
columns=pickle.load(open("task1/columns.pkl","rb"))
st.set_page_config(page_title="Restaurant Rating predictor",layout="centered")
st.title("Restaurant Rating predictor")
if "clear" not in st.session_state:
    st.session_state.clear=False
with st.form("form",clear_on_submit=True):

    st.subheader("Enter Restaurant Details")
    votes=st.number_input("Number of Votes",0,5000,0)
    cost=st.number_input("Average Cost for two",0,10000,0)
    price_range=st.slider("Price Range (1-4)",1,4,1)
    table_booking=st.selectbox("Table Booking Available?",["No","Yes"])
    online_delivery=st.selectbox("Online Deliverable Available?",["No","Yes"])
   
    submit=st.form_submit_button("Check Rating")
if submit:
        table_booking=1 if table_booking=="Yes" else 0
        online_delivery=1 if online_delivery=="Yes" else 0
        input_dict={'Votes':votes,'Average Cost for two':cost,'Price range':price_range,'Has Table booking':table_booking,'Has Online delivery':online_delivery}
        input_df=pd.DataFrame([input_dict])
        for col in columns:
            if col not in input_df.columns:
                input_df[col]=0
        input_df=input_df[columns]
        prediction=model.predict(input_df)[0]
        st.success(f"Predicted Rating:{round(prediction,2)}")
        if prediction>=4:
            st.info("Excellent Restaurant")
        elif prediction>=3:
            st.info("Good Restaurant")
        else:
            st.warning("Needs Improvement")
        
