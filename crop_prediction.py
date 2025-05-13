import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset

data = pd.read_csv('Crop_recommendation.csv')

x = data.drop('label',axis=1)
y = data['label']

# Split Data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)


# train model

model = RandomForestClassifier()
model.fit(x_train,y_train)

# Test accuracy

predic = model.predict(x_test)
print('accuracy :',accuracy_score(predic,y_test))


