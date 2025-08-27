import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import itertools


%matplotlib inline

import random
random.seed(42)

#data
data = pd.read_csv('car_sales_data.csv')

#cleaning up data stuff

data = data.drop(columns = ['Salesperson', 'Customer Name'])

data_target = data['Sale Price']
improved_data = data

improved_data.info(show_counts = True)



improved_data[['Year Sold', 'Month', 'Day']] = (data['Date'].str.split('-', expand=True)).astype(float)

improved_data['Year Sold'] = improved_data['Year Sold'].astype(float) + improved_data['Month'].astype(float) / 12

improved_data = improved_data.drop(columns = ['Month', 'Day', 'Date'])

improved_data.head()


#first data graph
grouped_data = data.groupby(['Car Model', 'Car Year', 'Year Sold'])

for (model, year, sold), group in grouped_data:
    if sold == 2023:
      plt.hist(group['Sale Price'], bins=80)

      plt.title(f"Sale Price Distribution for: {model}, Car Year:  {int(year)}, Sold in Year: {int(sold)}")
      plt.xlabel("Sale Price")
      plt.ylabel("Frequency")
      plt.show()


#another graph

grouped_data = data.groupby(['Car Model', 'Car Year'])['Sale Price'].mean().reset_index()

for model in grouped_data['Car Model'].unique():
    model_data = grouped_data[grouped_data['Car Model'] == model]
    plt.plot(model_data['Car Year'], model_data['Sale Price'])

    plt.xlabel("Car Year")
    plt.ylabel("Average Sale Price")
    plt.title(f"Average Sale Price of {model}")
    plt.show()


#another graph

grouped_data = data.groupby(['Car Model', 'Car Year'])['Sale Price'].mean().reset_index()

for model in grouped_data['Car Model'].unique():
    model_data = grouped_data[grouped_data['Car Model'] == model]
    plt.scatter(model_data['Car Year'], model_data['Sale Price'])

    plt.xlabel("Car Year")
    plt.ylabel("Average Sale Price")
    plt.title(f"Average Sale Price of {model}")
    plt.show()



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
dropped_data = improved_data.drop(columns = ['Car Make'])
transformed_data = dropped_data
transformed_data['Car Model'] = le.fit_transform(transformed_data['Car Model'])

transformed_data.info()




car_models = dropped_data['Car Model'].unique()

split_car_data = {}
for model in car_models:
  split_car_data[model] = dropped_data[dropped_data['Car Model'] == model]

print(split_car_data.keys())


corolla_data = grouped_data[grouped_data['Car Model'] == 'Corolla']


# Example with corolla and linear regression model, ill try to make it better
# Model was too inconsistant and too large to even want to reiterate

corolla_prepared_data = corolla_data['Car Year'].values
corolla_prepared_data = corolla_prepared_data.reshape(corolla_prepared_data.shape[0], 1)

data_target = corolla_data['Sale Price'].values
train, test, target, target_test = train_test_split(corolla_prepared_data, data_target, test_size = 0.3, random_state = 30)


#first attempt with a linear regression model
corolla_lin_reg = LinearRegression()

corolla_lin_reg.fit(train, target)

print("Predictions:", corolla_lin_reg.predict(test)[:5])
print("Actual labels:", list(target_test)[:5])

plt.ylim(29850, 30150)
plt.xlim(2005, 2030)

min_year = 2005
max_year = 2030
extended_years = np.linspace(min_year, max_year, 8)
extended_years = extended_years.reshape(extended_years.shape[0], 1)
price_preds = corolla_lin_reg.predict(extended_years)


plt.scatter(test, target_test, label='Test')
plt.scatter(train, target, label = 'Train')
plt.plot(extended_years, price_preds, color='red', label='Best Fit Line')
plt.xlabel('Car Year')
plt.ylabel('Sale Price')
plt.title('Best Fit Line for Corolla Prices')
plt.legend()
plt.show()

extended_years = np.linspace(2025, 2030, 5)
extended_years = extended_years.reshape(extended_years.shape[0], 1)
# Making best fit line

price_preds = corolla_lin_reg.predict(extended_years)

print("Estimated Average Price for Future Corollas: ", price_preds)




#getting rsme
preds = corolla_lin_reg.predict(test)
mse = mean_squared_error(target_test, preds)
rmse = np.sqrt(mse)
rmse

# Same thing but instead look at one car year and
# determine price depretiation over the years
grouped_data = improved_data.groupby(['Car Model', 'Car Year', 'Year Sold'])['Sale Price'].mean().reset_index()
civic_data = grouped_data[grouped_data['Car Model'] == 'Civic']

civic_data = civic_data[(civic_data['Car Year'] <= 2015) & (civic_data['Car Year'] >= 2013)]
# civic_data = civic_data[(civic_data['Car Year'] == 2015)]


civic_prepared_data = civic_data['Year Sold'].values
civic_prepared_data = civic_prepared_data.reshape(civic_prepared_data.shape[0], 1)
data_target = civic_data['Sale Price'].values


extended_years = np.linspace(2018, 2030, 10)
extended_years = extended_years.reshape(extended_years.shape[0], 1)

train, test, target, target_test = train_test_split(civic_prepared_data, data_target, test_size = 0.20, random_state = 30)

civic_lin_reg = LinearRegression()

civic_lin_reg.fit(train, target)

price_preds = civic_lin_reg.predict(extended_years)

print("Predictions:", civic_lin_reg.predict(test)[:5])
print("Actual labels:", list(target_test)[:5])



#anouther data graph
plt.ylim(29500, 30500)
plt.xlim(2018, 2030)

plt.scatter(test, target_test, label='Test')
plt.scatter(train, target, label = 'Train')

plt.plot(extended_years, price_preds, color='red', label='Best Fit Line')
plt.xlabel('Car Year')
plt.ylabel('Sale Price')
plt.title('Best Line for Civics older than 2015 Price')
plt.legend()
plt.show()


#same thing again a graph
plt.ylim(29000, 31000)
plt.xlim(2018, 2030)

plt.scatter(test, target_test, label='Test')
plt.scatter(train, target, label = 'Train')

plt.plot(extended_years, price_preds, color='red', label='Best Fit Line')
plt.xlabel('Car Year')
plt.ylabel('Sale Price')
plt.title('Best Line for Civics older than 2015 Price')
plt.legend()
plt.show()

print("In ", int(extended_years[9]), " the expected price of a Civic older than 2017 is: $", int(price_preds[9]))

