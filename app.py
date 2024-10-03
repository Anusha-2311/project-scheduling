############## Libraries/Modules ##############

# Basic Libraries
import pandas as pd
import numpy as np

# ModelLoading Libraries
import joblib

# UI & Logic Library
import streamlit as st

####################### Loading Trained Model Files #########
ohe = joblib.load("ohe.pkl") # converting categorical x to numerical
sc = joblib.load("sc.pkl") # converting numeric cols under one scale
#poly = joblib.load("poly.pkl") # convert x features to poly features
model = joblib.load("construction_rfr.pkl") # trained poly regression file

########################## UI Code ################################

st.header("Project Scheduling Estimation")

# Dividing Row into columns in streamlit window
p1, p2, p3 = st.columns(3)

with p2:
    st.image("construction.jpg")

st.write("This app built on the below features to estimate the project completion days.")

df = pd.read_csv("Timeline_Estimation_final_data(1).csv")

st.dataframe(df.head(5))

st.subheader("Enter Project Details to Estimate the Completion Days:")



# Form Type Input
col1, col2, col3, col4= st.columns(4)
with col1:
    unnamed = st.number_input(" unnamed:")
with col2:
    pro_type = st.selectbox("Project Type:", df['Project Type'].unique())
with col3:
    pro_size = st.number_input("project size (sq ft):")
with col4:
    Budget = st.number_input("Budget in INR:")


col5, col6, col7, col8 = st.columns(4)
with col5:
    Man_power = st.number_input("no.of workers:")
with col6:
    Subcontractors = st.number_input("subcontractors invovled:")
with col7:
    No_of_Tasks = st.number_input("No_of_Tasks:")
with col8:
    Task_Completion_Percentage = st.number_input("Task_Completion_Percentage:")


col9, col10, col11,col12 = st.columns(4)
with col9:
    no_of_delays = st.number_input("no.of.delays:")
with col10:
    weather_condition = st.selectbox("Weather Conditions:", df['Weather Conditions'].unique())
with col11:
    Material_Availability = st.selectbox("Material Availability:", df['Material Availability'].unique())
with col12:
    Permits_Regulatory_Approvals = st.number_input("Permits Regulatory Approvals:")


col13, col14, col15, col16 = st.columns(4)
with col3:
    Change_Orders = st.number_input("Change Orders:")
with col4:
    Site_Location = st.selectbox("Site Location:", df['Site Location'].unique())
with col5:
    Pro_Complexity = st.number_input("Project Complexity (1 -10):")
with col6:
    Pro_Manager_Exp = st.number_input("Project Manager Experience (years):")

col17, col18, col19 = st.columns(3)
with col7:
    Contract_Type = st.selectbox("Contract Type:", df['Contract Type'].unique())
with col8:
    Risk_Factors = st.selectbox("Risk Factors:", df['Risk Factors'].unique())
with col19:
    Pre_Similar_Project_Performance = st.number_input("Pre Similar Project Performance:")


###################### Logic Code #############################

if st.button("Completion Days"):

    row = pd.DataFrame([[unnamed,pro_type,pro_size,Budget,Man_power,Subcontractors,No_of_Tasks,Task_Completion_Percentage,
                         no_of_delays,weather_condition,Material_Availability,Permits_Regulatory_Approvals,
                        Change_Orders,Site_Location,Pro_Complexity,Pro_Manager_Exp,Contract_Type,
                        Risk_Factors,Pre_Similar_Project_Performance]], columns=df.columns)
    row = row.drop(["Unnamed: 0"],axis=1)
    
    print()
    st.write("Given Input Data:")
    
    st.dataframe(row)
    print()
    
    # Applying Feature Modification steps before giving it to model
    
    ## Binary encoding ##
    row['Material Availability'].replace({'No':0, 'Yes':1},inplace = True)
    row['Contract Type'].replace({'Time and materials':0, 'Fixed price':1},inplace = True)
    
    ## Ordinal encoding
    row['Weather Conditions'].replace({'Moderate':0, 'Favorable':1, 'Adverse':2},inplace = True)
    row['Risk Factors'].replace({'Low':0, 'Medium':1, 'High':2},inplace = True)

    
    # Onehot Encoding
    row_ohe = ohe.transform(row[["Project Type","Site Location"]]).toarray()
    row_ohe = pd.DataFrame(row_ohe, columns=ohe.get_feature_names_out())
    
    row=row.drop('Project Type',axis=1)
    row=row.drop('Site Location',axis=1)
    row=pd.concat([row, row_ohe], axis=1)
    

    # Scaling
    row.iloc[:, :] = sc.transform(row)

    
    
    schedule = round(model.predict(row)[0])
    
    st.write(f" Time Line Estimation: {schedule} Days ")

    

