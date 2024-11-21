1. Import Libraries
The script begins by importing necessary libraries for data analysis, visualization, and machine learning.
•	Libraries used: 
o	numpy: For numerical operations.
o	pandas: For data manipulation and analysis.
o	warnings: To suppress unnecessary warnings.
o	seaborn, matplotlib.pyplot: For data visualization.
o	plotly.express: For interactive visualizations.
o	sklearn: For data preprocessing and building machine learning models.
o	pydot: For visualizing decision trees.


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
________________________________________
2. Data Importing
The dataset churn.csv is loaded into a pandas DataFrame.
•	Code: 
•	telcom = pd.read_csv(r"C:\Users\naveen kumar pandey\Desktop\TEST FILE\churn.csv")
•	telcom.head()

•	Purpose: To explore the first five rows and understand the dataset structure, including its columns and sample data.
________________________________________
3. Data Preprocessing
3.1 Drop Irrelevant Columns
The column customerID is removed because it does not provide meaningful information for churn prediction.

•	Code: 
•	telcom = telcom.drop('customerID', axis=1)

3.2 Handle Missing Values
The column TotalCharges contains spaces, which are replaced with NaN values. Missing values are filled with the column's mean.
•	Code: 
•	telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ", np.nan).astype(float)
•	telcom.TotalCharges.fillna(telcom.TotalCharges.mean(), inplace=True)

3.3 Data Type Conversion
The SeniorCitizen column, originally numerical (1 for Yes, 0 for No), is converted into categorical data.
•	Code: 
•	telcom.SeniorCitizen = telcom.SeniorCitizen.replace({1: "Yes", 0: "No"})

3.4 Replace Redundant Values
Columns like OnlineSecurity, DeviceProtection, etc., contain No internet service, which is replaced with No. Similarly, No phone service in MultipleLines is replaced with No.
•	Code: 
•	telcom.OnlineSecurity = telcom.OnlineSecurity.replace({'No internet service': 'No'})
•	telcom.MultipleLines = telcom.MultipleLines.replace({'No phone service': 'No'})
________________________________________
4. Outlier Identification
Box plots are created to identify potential outliers in TotalCharges, MonthlyCharges, and tenure.
•	Code: 
•	sns.boxplot(y=telcom.TotalCharges)
•	sns.boxplot(y=telcom.MonthlyCharges)
•	sns.boxplot(y=telcom.tenure)
________________________________________
5. Feature Engineering
5.1 Create Tenure Groups
Customers are grouped into tenure categories based on the number of months they've stayed with the company.
•	Code: 
•	def tenure_lab(telcom):
•	    if telcom["tenure"] <= 12:
•	        return "Tenure_0-12"
•	    elif 12 < telcom["tenure"] <= 24:
•	        return "Tenure_13-24"
•	    elif 24 < telcom["tenure"] <= 48:
•	        return "Tenure_25-48"
•	    elif 48 < telcom["tenure"] <= 60:
•	        return "Tenure_49-60"
•	    else:
•	        return "Tenure_gt_60"
•	
•	telcom["tenure_group"] = telcom.apply(lambda x: tenure_lab(x), axis=1)
5.2 Encode Categorical Data
Label encoding is applied to convert categorical columns into numerical values.
•	Code: 
•	from sklearn.preprocessing import LabelEncoder
•	telcom_dummies = telcom.select_dtypes(include=['object']).apply(LabelEncoder().fit_transform)
5.3 Combine Processed Data
Numerical and encoded categorical data are concatenated into a final dataset for model building.
•	Code: 
•	telcom = pd.concat([telcom.select_dtypes(include=[np.number]), telcom_dummies], axis=1)
________________________________________
6. Data Splitting
The dataset is split into training (70%) and testing (30%) sets.
•	Code: 
•	from sklearn.model_selection import train_test_split
•	
•	X = telcom.drop('Churn', axis=1)
•	Y = telcom[['Churn']]
•	
•	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1234)
________________________________________
7. Model Building: Random Forest
A Random Forest Classifier is built to predict customer churn.
•	Key Parameters:
o	n_estimators=25: Number of decision trees in the forest.
o	max_depth=4: Maximum depth of each tree.
o	min_samples_split=100: Minimum samples required to split an internal node.
o	min_samples_leaf=50: Minimum samples required to be a leaf node.
•	Code:
•	from sklearn.ensemble import RandomForestClassifier
•	
•	RFModel = RandomForestClassifier(random_state=20,
•	                                 n_estimators=25,
•	                                 criterion="gini",
•	                                 max_depth=4,
•	                                 min_samples_split=100,
•	                                 min_samples_leaf=50,
•	                                 max_features="sqrt")
•	
•	RFModel.fit(X_train, y_train)
________________________________________
8. Feature Importance
The importance of each feature is calculated and visualized.
•	Code: 
•	imp = pd.Series(data=RFModel.feature_importances_, index=RFModel.feature_names_in_).sort_values(ascending=False)
•	sns.barplot(y=imp.head().index, x=imp.head().values, palette="BrBG")
________________________________________
9. Model Evaluation
Predictions are made on the training data to evaluate the model's performance.
•	Code: 
•	train['Predicted'] = RFModel.predict(X_train)
________________________________________
10. Decision Tree Visualization
One of the trees in the Random Forest is visualized using pydot and sklearn.
•	Code: 
•	from sklearn.tree import export_graphviz
•	import pydot
•	
•	tree = RFModel.estimators_[4]
•	export_graphviz(tree, out_file='abc.dot',
•	                feature_names=list(X.columns),
•	                class_names=['No', 'Yes'],
•	                rounded=True,
•	                filled=True)
________________________________________
11. Interactive Visualization
A pie chart of churn distribution is created using Plotly.
•	Code: 
•	import plotly.express as px
•	
•	fig = px.pie(telcom, names='Churn', color='Churn',
•	             color_discrete_map={'Yes': 'red', 'No': 'green'})
•	fig.show()
________________________________________
Conclusion
The script effectively preprocesses the dataset, performs feature engineering, builds a Random Forest model to predict customer churn, and visualizes key insights. The model can now be further tuned or deployed to improve churn predictions.

