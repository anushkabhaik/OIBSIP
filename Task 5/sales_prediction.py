import pandas as pd
sales = pd.read_csv('Advertising.csv')
sales.head()
sales.isnull().sum()
sales.describe()

from sklearn.preprocessing import MinMaxScaler
features = ['TV','Radio','Newspaper','Sales']
A = sales[features]
scaler = MinMaxScaler()
A_norm = scaler.fit_transform(A)
A_norm_sales = pd.DataFrame(A_norm,columns=features)
A_norm_sales.head()
sales[features]=A_norm_sales
sales.head()

from sklearn.model_selection import train_test_split
X = sales.drop(columns=['Sales'])
y = sales['Sales']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=50)

from sklearn.ensemble import RandomForestRegressor
pred_model = RandomForestRegressor(n_estimators=100,random_state=50)
pred_model.fit(X_train,y_train)
y_pred=pred_model.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mse = ",mse)
print("r2_score = ",r2)







