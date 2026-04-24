import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('gb_model.pkl','rb'))

scaler=StandardScaler()

st.title("insurance price prediction app")

age=st.number_input('Age',min_value=1,max_value=100,value=20)
gender=st.selectbox('gender',('male','female'))
bmi=st.number_input('Bmi',min_value=10,max_value=80,value=20)
smoker=st.selectbox('smoking',('yes','no'))
children=st.number_input('Number of Children',min_value=0,max_value=10,value=2)
region=st.selectbox('region',('southwest','southeast','northwest','northeast'))

#encode
Smoker=1 if  smoker=='yes' else 0
sex_female = 1 if gender=='female' else 0
sex_male = 1 if gender=='male' else 0
region_dict={'southeast':3,'northeast':2,'northwest':1,'southwest':0}
Region=region_dict[region]
# create dataframe
input_features=pd.DataFrame({
    'age':[age],
    'bmi':[bmi],
    'children':[children],
    'Smoker':[Smoker],
    'sex_female':[sex_female],
    'sex_male':[sex_male],
    'Region':[Region]
})
input_features[['age','bmi']]=scaler.fit_transform(input_features[['age','bmi']])
if st.button('predict'):
  predictions=model.predict(input_features)
  output=round(np.exp(predictions[0]),2)
  st.success(f'price prediction: ${output}')