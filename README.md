#  Crop Recommendation System

This project is a machine learning model that recommends the most suitable crop to grow based on agricultural features such as soil nutrients and environmental conditions. It uses a **Random Forest Classifier** to predict crop types based on input data.

---

##  Dataset

The dataset used is `Crop_recommendation.csv`, which contains the following features:

- `N` - Nitrogen content in soil
- `P` - Phosphorous content in soil
- `K` - Potassium content in soil
- `temperature` - Temperature in degree Celsius
- `humidity` - Relative humidity in %
- `ph` - pH value of the soil
- `rainfall` - Rainfall in mm
- `label` - Recommended crop to grow

---

##  Libraries Used

- `pandas`
- `scikit-learn`:
  - `train_test_split`
  - `RandomForestClassifier`
  - `accuracy_score`

---

##  Model Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('Crop_recommendation.csv')

# Separate features and labels
x = data.drop('label', axis=1)
y = data['label']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Test accuracy
predic = model.predict(x_test)
print('accuracy :', accuracy_score(predic, y_test))


##  Sample Output

```text
accuracy : 0.9806818181818182
