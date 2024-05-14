import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("RidingMowers.csv")

# Sidebar for user input
st.sidebar.title("KNN Classifier")

# Load data
df = load_data()

# Main content
st.title("KNN Classifier")

# Display the dataset
if st.checkbox("Show Dataset"):
    st.write(df)

# Preprocessing
dum_df = pd.get_dummies(df, drop_first=True)
X = dum_df.iloc[:, :-1]
y = dum_df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021, stratify=y)

# Model training
k = st.sidebar.slider("Select number of neighbors (k)", 1, 20, 5)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Model evaluation
y_pred_prob = knn.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
st.write(f"ROC AUC Score: {roc_auc}")

# Prediction
if st.sidebar.button("Predict"):
    # Your prediction code here
    pass
