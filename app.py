import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import  accuracy_score,confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("RidingMowers.csv")
dum_df = pd.get_dummies(df,drop_first=True)
X = dum_df.iloc[:,:-1]
y = dum_df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2021,
                                                    stratify=y)
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)

y_pred_prob = knn.predict_proba(X_test)

y_pred = knn.predict(X_test)

print(roc_auc_score(y_test, y_pred_prob[:,1]))

