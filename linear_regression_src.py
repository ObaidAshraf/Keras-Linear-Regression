from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("brain-body.csv", delimiter=",")

X = data[['BrainWeight']].values
Y = data[['BodyWeight']].values

model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(Adam(lr=0.8), 'mean_squared_error')
model.fit(X, Y, epochs=50)

Y_predict = model.predict(X)

data.plot(kind='scatter',
          x = 'BrainWeight',
          y = 'BodyWeight',
          title='Brain Weight and Body Weight Regression Curve')

plt.plot(X, Y_predict, color='red', linewidth=3)
plt.show()
