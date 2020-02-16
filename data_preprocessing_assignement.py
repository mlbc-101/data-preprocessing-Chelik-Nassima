""" Do not delete this section, Please Commit your changes after implementing the necessary code.

- The data file called Social_Network_Ads.csv.
- Your Job is to preprocess this data because we gonna use it later one in the course.

The Features of this dataset are:
	- UserID: Which represent id of user in the database.
	- Gender: Can be male or female.
	- EstimatedSalary: The salary of the user.
	- Purchased: An integer number {1 if the user purshased something, 0 otherwise}
	
	The target variable for this data is the purshased status.

Happy coding."""

# Step 0: import the necessary libraries: pandas, matplotlib.pyplot, and numpy.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Step 1: load your dataset using pandas
data= pd.read_csv("Social_Network_Ads.csv")

# Step 2: Handle Missing data if they exist.

# Step 3: Encode the categorical variables.
from sklearn.preprocessing import LabelEncoder 
label_x = LabelEncoder()
data['Gender']  = label_x.fit_transform(data['Gender'] )

# Step 4: Do Feature Scaling if necessary.
x=data.iloc[:,1:4]
y=data['Purchased']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x= sc.fit_transform(x)
# Final Step: Train/Test Splitting.
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y,test_size = 0.3 , random_state = 2424)