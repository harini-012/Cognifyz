
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
df=pd.read_csv(r"D:\Cognifyz\Task1\Dataset.csv")
print("Data Shape:",df.shape)
print(df.head())
df.drop(['Restaurant ID','Restaurant Name','Address','Locality','Locality Verbose'],axis=1,inplace=True,errors='ignore')
binary_cols=['Has Table booking', 'Has Online delivery','Is delivering now','Switch to order menu']
for col in binary_cols:
    if col in df.columns:
        df[col]=df[col].map({'Yes':1,'No':0})
df.fillna(df.mean(numeric_only=True),inplace=True)
plt.figure()
df['Aggregate rating'].hist(bins=20)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("count")
plt.show()
df.drop(['Rating color','Rating text'],axis=1,inplace=True,errors='ignore')
df=pd.get_dummies(df,columns=['City','Cuisines','Currency'],drop_first=True)
X=df.drop('Aggregate rating',axis=1)
y=df['Aggregate rating']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Mean Squared Error:",mean_squared_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))
importance=pd.Series(model.feature_importances_,index=X.columns)
importance=importance.sort_values(ascending=False)
print("\nTop Features:\n",importance.head(10))
plt.figure()
importance.head(10).plot(kind='barh')
plt.title("Top 10 features")
plt.show()
pickle.dump(model,open("model.pkl","wb"))
pickle.dump(X.columns.tolist(),open("columns.pkl","wb"))
print("Model saved successfully")