import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

df = pd.read_csv("car.data")

df.head()
df.shape
df.info()
df.isnull().sum()
df.describe()

duplicate = df[df.duplicated()]
duplicate

ax = sns.countplot(x='class', data=df, palette='Set1')
ax.set_title("Distribution of class")
plt.show()

f, ax = plt.subplots(figsize=(5,4))
ax = sns.countplot(x='class', hue= 'buying', data=df, palette='Set1' )
ax.set_title("Frequency Distribution of class wrt buying")
plt.show()

f, ax = plt.subplots(figsize=(5,4))
ax = sns.countplot(x='class', hue= 'maint', data=df, palette='Set1' )
ax.set_title("Frequency Distribution of class wrt maint")
plt.show()

f, ax = plt.subplots(figsize=(5,4))
ax = sns.countplot(x='class', hue= 'doors', data=df, palette='Set1' )
ax.set_title("Frequency Distribution of class wrt doors")
plt.show()

f, ax = plt.subplots(figsize=(5,4))
ax = sns.countplot(x='class', hue= 'lug_boot', data=df, palette='Set1' )
ax.set_title("Frequency Distribution of class wrt lug_boot")
plt.show()

f, ax = plt.subplots(figsize=(5,4))
ax = sns.countplot(x='class', hue= 'safety', data=df, palette='Set1' )
ax.set_title("Frequency Distribution of class wrt safety")
plt.show()


encoder = ce.OrdinalEncoder(cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
df = encoder.fit_transform(df)

x = df.drop("class",axis = 1)
y = df["class"]

X_train, X_test, Y_train, Y_test =train_test_split(x, y, test_size=0.2,random_state=42 )

cols = X_train.columns
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_train = pd.DataFrame(X_train, columns= [cols])
X_test = pd.DataFrame(X_test, columns=[cols])

from sklearn  import svm
clf = svm.SVC()
clf.fit(X_train, Y_train)

Y_pred =clf.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(Y_test, Y_pred)
print(np.abs(score)*100)



