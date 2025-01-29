import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

file_path = r'C:\Users\dell\.vscode\New folder\Problem 3.csv'
data = pd.read_csv(file_path)

filtered_data = data[['MaxTemp', 'MinTemp']].dropna()

filtered_data['MaxTemp'] = pd.to_numeric(filtered_data['MaxTemp'], errors='coerce')
filtered_data['MinTemp'] = pd.to_numeric(filtered_data['MinTemp'], errors='coerce')
filtered_data = filtered_data.dropna()


X = filtered_data['MinTemp'].values.reshape(-1, 1)
y = filtered_data['MaxTemp'].values


model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=filtered_data['MinTemp'], y=filtered_data['MaxTemp'], alpha=0.5, label='Actual Data')
plt.plot(filtered_data['MinTemp'], y_pred, color='red', label='Regression Line')
plt.title('Relationship Between Minimum and Maximum Temperatures')
plt.xlabel('Minimum Temperature (°C)')
plt.ylabel('Maximum Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

slope, intercept = model.coef_[0], model.intercept_
print(f"Equation of the line: MaxTemp = {slope:.3f} * MinTemp + {intercept:.3f}")