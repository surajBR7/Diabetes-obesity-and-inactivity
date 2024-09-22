import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import *
from sklearn.model_selection import ShuffleSplit,learning_curve
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

excel = pd.ExcelFile("cdc-diabetes-2018.xlsx")

sheetName = excel.sheet_names
dataframes = []

for i in sheetName:
	df = excel.parse(i)
	dataframes.append(df)

df_d = dataframes[0]
df_o = dataframes[1]
df_i = dataframes[2]

data_dict_list = []

for index, row in df_d.iterrows():
    row_dict = {
        "FIPS": row["FIPS"],
        "COUNTY": row["COUNTY"],
        "STATE": row["STATEW"],
        "YEAR": row["YEAR"]
    }
    data_dict_list.append(row_dict)
df_d.drop(columns=["YEAR", "COUNTY", "STATEW"], inplace=True)

# for data_dict in data_dict_list:
#     print(data_dict)

print("Length of dataframes df_d -> ",len(df_d))
print("Length of dataframes df_o -> ",len(df_o))
print("Length of dataframes df_f -> ",len(df_i))

merge_df_do = pd.merge(df_d, df_o, on='FIPS',how='inner')
print("Length of dataframes D-O -> ",len(merge_df_do))

merge_df_do.drop(columns=["YEAR", "COUNTY", "STATE"], inplace=True)

de_new_f = df_i.rename({'FIPDS': 'FIPS'}, axis='columns')
merge_df_all = pd.merge(merge_df_do, de_new_f, on='FIPS',how='inner')
print("Length of dataframes D-O-F -> ",len(merge_df_all))

merge_df_all.drop(columns=["YEAR", "COUNTY", "STATE"], inplace=True)
print(merge_df_all.head().to_string())

print(merge_df_all.info())
print(merge_df_all.describe())

print(merge_df_all.isnull())
print(merge_df_all.isnull().sum())

print(sns.pairplot(merge_df_all.iloc[:, -3:]))
plt.show()

col = merge_df_all.iloc[:, -3:].columns
print(col)

fig, ax = plt.subplots(nrows = 1, ncols=3, figsize = (16,4))

for i in range(3):
	sns.histplot(merge_df_all[col[i]], ax=ax[i], kde=True)
		# print(merge_df_all[col])
plt.tight_layout()
plt.show()

skewness = merge_df_all.iloc[:, -3:].skew()
print(skewness)
for column, skew in skewness.items():
	print(f'Skewness for {column}: {skew:.2f}')

correlation_Mat = merge_df_all.iloc[:, -3:].corr()
print(correlation_Mat)

axis_corr = sns.heatmap(
correlation_Mat,
vmin=-1, vmax=1, center=0,
annot = True,
annot_kws = {'size': 12},
)
cbar = axis_corr.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
plt.show()

print(correlation_Mat.index)

print("\n-------------------------------------- Dia - Obe --------------------------------------\n")

X = merge_df_all[['% OBESE']]
y = merge_df_all[['% DIABETIC']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_predict = model.predict(X_test)

plt.scatter(X_test, y_test, color="blue", label="Actual Data")
plt.plot(X_test, y_predict, color="black", linewidth=3, label="Regression Line")
plt.xlabel(" % OBESE ", fontsize=10)
plt.ylabel(" % DIABETIC ", fontsize=10)
plt.title("Scatter Plot with Regression Line", fontsize=14)
plt.legend()
plt.grid(True) 
plt.tight_layout()
plt.show()

score = r2_score(y_test,y_predict)
print(score)

print("\n-------------------------------------- Dia - Ina --------------------------------------\n")

X = merge_df_all[['% INACTIVE']]
y = merge_df_all[['% DIABETIC']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_predict = model.predict(X_test)

plt.scatter(X_test, y_test, color="blue", label="Actual Data")
plt.plot(X_test, y_predict, color="black", linewidth=3, label="Regression Line")
plt.xlabel(" % INACTIVE ", fontsize=10)
plt.ylabel(" % DIABETIC ", fontsize=10)
plt.title("Scatter Plot with Regression Line", fontsize=14)
plt.legend()
plt.grid(True) 
plt.tight_layout()
plt.show()

score = r2_score(y_test,y_predict)
print("R square value ",score)

mean_y = np.mean(y_test)
tss = np.sum((y_test - mean_y) ** 2)
rss = np.sum((y_test - y_predict) ** 2)
r_squared = 1 - (rss / tss)
print("R-squared:", r_squared)

print("\n-------------------------------------- Dia - Ina & Obe --------------------------------\n")

import statsmodels.formula.api as smf
from statsmodels.compat import lzip
import statsmodels.stats.api as sms

X = merge_df_all[['% OBESE','% INACTIVE']]
y = merge_df_all[['% DIABETIC']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_predict = model.predict(X_test)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_test['% OBESE'], X_test['% INACTIVE'], y_test, c='blue', marker='o', label='Actual Data')
# ax.scatter(X_test['% OBESE'], X_test['% INACTIVE'], y_predict, c='red', marker='x', label='Predicted Values')

# ax.set_xlabel('% OBESE')
# ax.set_ylabel('% INACTIVE')
# ax.set_zlabel('% DIABETIC')
# ax.set_title("Scatter Plot with Multilinear Regression")

# ax.legend()
# plt.tight_layout()
# plt.show()

score = r2_score(y_test,y_predict)
print("R square value ",score)

mean_y = np.mean(y_test)
tss = np.sum((y_test - mean_y) ** 2)
rss = np.sum((y_test - y_predict) ** 2)
r_squared = 1 - (rss / tss)

print("R-squared:", r_squared)

score = r2_score(y_test,y_predict)
mae = mean_absolute_error(y_test,y_predict)
mse = mean_squared_error(y_test,y_predict)
rmse = np.sqrt(mse)

print("R2 score: ", score)
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)

residuals = y_test - y_predict

# Convert residuals to a DataFrame
residuals_df = pd.DataFrame(residuals, columns=['Residuals'])

# Manually add a constant (intercept) to the independent variables DataFrame
X_test_with_const = sm.add_constant(X_test)

# Perform the Breusch-Pagan test
_, p_value, _, _ = het_breuschpagan(residuals, X_test_with_const)

print("Breusch-Pagan Test p-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("Heteroscedasticity is present (reject the null hypothesis)")
else:
    print("Heteroscedasticity is not present (fail to reject the null hypothesis)")


rows = 2
cols = 2
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize = (16, 4))
ax[0, 0].set_title("FIPS")
ax[0, 1].set_title("DIABETIC")
ax[1, 0].set_title("OBESE")
ax[1, 1].set_title("INACTIVE")
col = merge_df_all.columns
index = 0

for i in range(rows):
    for j in range(cols):
        sns.regplot(x = merge_df_all[col[index]], y = merge_df_all['% DIABETIC'], ax = ax[i][j])
        index = index + 1
plt.tight_layout()
plt.show()

cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

mse_scores = -cross_val_scores

mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

print("Cross-Validation Mean MSE:", mean_mse)
print("Cross-Validation Std MSE:", std_mse)