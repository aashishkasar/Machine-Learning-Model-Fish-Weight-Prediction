import pandas as pd
data=pd.read_csv(r"D:\Data Sets\Fish.csv")
data
data['Species'].value_counts()
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
data['Species']=l.fit_transform(data['Species'])
Species=pd.DataFrame(data[['Species']])
Species
Species=pd.DataFrame(data[['Species']])
Species
data
cl=data[['Weight']]
fv=data[['Species','Length1','Length2','Length3','Height','Width']]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(fv,cl,train_size=0.7)

lr=LinearRegression()

model=lr.fit(x_train,y_train)

lr.get_params()

predi=model.predict(x_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

mean_squared_error(y_test,predi)

mean_absolute_error(y_test,predi)

r2_score(y_test,predi)

model.coef_

model.intercept_

y=(9491*1.1)+26740
y

len(model.predict(fv))

model.predict(fv).reshape(len(model.predict(fv)))

import joblib

joblib.dump(model,r"D:\Data Sets\ML Models\fish1.joblib")


