import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load data
housing= pd.read_csv("housing.csv")

# 2. Create a stratified test set on the base of income categrory
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1 ,test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop('income_cat', axis=1)
    strat_test_set = housing.loc[test_index].drop('income_cat', axis=1)

housing = strat_train_set.copy()

# 3. Separate predictors and labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis=1)

# 4. Separate numerical and categorical value
num_attribs = housing.drop(['ocean_proximity'], axis=1).columns.tolist()
cat_attribs = ['ocean_proximity']

# 5. Pipeline 
# Numerical Pipeline
num_pipeline = Pipeline([
    ("imputer" , SimpleImputer(strategy='median')),
    ("scaler" , StandardScaler()),
])

# Categorical Pipeline
cat_pipeline = Pipeline([
    ("onehot" , OneHotEncoder(handle_unknown="ignore")),
])

# Full Pipeline 
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

# 7. Train the model

# Linear Regressor Model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_pred = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_pred)
# print(f'1 Linear: {lin_rmse}')
lin_rmse = -cross_val_score(lin_reg, housing_prepared,housing_labels, scoring= "neg_root_mean_squared_error", cv=10)
print(pd.Series(lin_rmse).describe())

# Decision Tree Model
dec_reg = DecisionTreeRegressor(random_state=42)
dec_reg.fit(housing_prepared, housing_labels)
dec_pred = dec_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels, dec_pred)
# print(f'2 Tree: {dec_rmse}')
dec_rmse = -cross_val_score(dec_reg, housing_prepared,housing_labels, scoring= "neg_root_mean_squared_error", cv=10)
print(pd.Series(dec_rmse).describe())

# Random Forest Model
ran_reg = RandomForestRegressor(random_state=42)
ran_reg.fit(housing_prepared, housing_labels)
ran_pred = ran_reg.predict(housing_prepared)
# ran_rmse = root_mean_squared_error(housing_labels, ran_pred)
# print(f'1 Forest: {ran_rmse}')
ran_rmse = -cross_val_score(ran_reg, housing_prepared,housing_labels, scoring= "neg_root_mean_squared_error", cv=10)
print(pd.Series(ran_rmse).describe())