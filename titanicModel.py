#To simplify things even further, you will only use four variables: age, sex, embarked, and survived where survived is the class label.
# Import dependencies
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump

def createFeatures(df, model_columns=[]):
	categoricals = []
	for col, col_type in df.dtypes.iteritems():
		if col_type == 'O':
			categoricals.append(col)
		else:
			df[col].fillna(0, inplace=True)

	# convert categorical to numeric ones
	df_ohe = pd.get_dummies(df, columns=categoricals, dummy_na=True)

	# load original columns
	if model_columns != []:
		df_ohe = df_ohe.reindex(columns=model_columns, fill_value=0)

	return df_ohe

dataPath = 'resources/train.csv'
df = pd.read_csv(dataPath)

include = ['Age', 'Sex', 'Embarked', 'Survived'] # Only four features
df = df[include]

#create features
df_ohe = createFeatures(df)

#train a logistic regression
dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)

#persist the column names
model_columns = list(x.columns)
dump(model_columns, 'models/model_columns.pkl')

#save the model into file model.pkl in folder models
dump(lr, 'models/model.pkl')