import numpy as np
import pandas as pd 
import pickle
import streamlit as st

#Storing Columns for Encoding and Scaling
con_clos=["age","cigsPerDay","totChol","sysBP","diaBP","BMI","heartRate","glucose"] #the columns that need to be scaled
cat_cols=["Sex","diabetes"] #the columns that need to be encoded

# Loading Stored Objects for Reuse
cols=pickle.load(open(r"updated-heart-diseases-detection/fetures.pkl","rb"))
scaller=pickle.load(open(r"updated-heart-diseases-detection/scaller.pkl","rb"))
encoder=pickle.load(open(r"updated-heart-diseases-detection/encoders.pkl","rb"))
models=pickle.load(open(r"updated-heart-diseases-detection/models.pkl","rb"))
pca=pickle.load(open(r"updated-heart-diseases-detection/PCA.pkl","rb"))




# Method for Predicting Heart Disease
def detect_heart_disease(features,model):
    data=pd.DataFrame(features,columns=cols)
    for i in cat_cols:
        data[i]=encoder[i].transform([data[i]])
    data[con_clos]=scaller.transform(data[con_clos])
    if model=="SVC":
        predected=models[model].predict(data)
    else :
        pca_data=pca.transform(data)
        predected=models[model].predict(pca_data)

    return predected

# Starting Streamlit Code for Heart Disease Prediction

st.title('heart disease detiction')  # Adding a Title to the Streamlit Page

# Collecting User Inputs Using Streamlit
model=st.selectbox("Please select the model you would like to use for predicting heart disease:" , ["SVC","XGBoost"])
sex=st.selectbox("your gender is :",["male","female"])
Age=st.slider("your age is : ",min_value=25,max_value=75,value=49)
cigs_per_day =st.slider("How many cigarettes do you smoke per day?",min_value=0,max_value=70,value=8)
BPMeds =st.selectbox("Have you ever taken blood pressure medication (BPMeds) in your life ? ",[0,1])
prevalentStrock =st.selectbox("Have you ever had a stroke ? ",[0,1])
prevalentHyp=st.selectbox("Do you have prevalent hypertension (high blood pressure)?",[0,1])
diabetes=st.selectbox("Do you have diabetes?",["Yes","No"])
totChol=st.slider("What is your total cholesterol level?",min_value=100,max_value=460,value=230)
sysBP=st.slider("What is your systolic blood pressure?",min_value=78,max_value=300,value=130)
biaBP=st.slider("What is your diastolic blood pressure?",min_value=40,max_value=150,value=82)
BMI=st.slider("What is your Body Mass Index ?",min_value=10,max_value=55,value=25)
heartRate=st.slider("What is your heart rate?",min_value=40,max_value=150,value=75)
glucose=st.slider("What is your glucose level?",min_value=35,max_value=400,value=80)

# Saving Collected Variables in a List for Prediction
features=[[sex,Age,cigs_per_day,BPMeds,prevalentStrock,prevalentHyp,diabetes,
           totChol,sysBP,biaBP,BMI,heartRate,glucose]]


# Adding a Button for Interactive Predictions in Streamlit
if st.button("predict"):
    result=detect_heart_disease(features,model)
    if result==0:
        st.write("According to the prediction, you do not have heart disease.")
    else:
        st.write("According to the prediction, you have heart disease.")


# End of Streamlit interface
print("\nThank you for using the Heart Disease Prediction model. 😊")
print("Your health is important, and this model can help you understand your risk.")
print("Please make sure to consult a healthcare professional for further insights and recommendations!")
print("\nStay healthy and happy! 😊")
