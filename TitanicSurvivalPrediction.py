import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')

# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].map({'female':'1','male':'0'})

# Fill the nan values in the age column
# We give value a mean age number to fill nans
passengers['Age'].fillna(value=passengers['Age'].mean(),inplace=True)



# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)


# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)
print (passengers)

# Select the desired features
features = passengers[['Sex','Age','FirstClass','SecondClass']]
survival = passengers[['Survived']]

# Perform train, test, split
X_train, X_test, y_train, y_test = train_test_split(features,survival,test_size = 0.2)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression()
model.fit(X_train,y_train)

# Score the model on the train data
print(model.score(X_train,y_train))

# Score the model on the test data
print(model.score(X_test, y_test))

# Analyze the coefficients
print(model.coef_)
print(list(zip(['Sex','Age','FirstClass','SecondClass'],model.coef_[0])))
# As you can see Sex is the main feature to survive, we can check later in predictions

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,60.0,1.0,0.0])
You = np.array([0.0,42,1.0,0.0])

# Combine passenger arrays
sample_passengers = np.array([ Jack , Rose, You ])


# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)

# Make survival predictions!
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))
