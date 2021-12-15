import matplotlib
matplotlib.use("TkAgg")
from utils import preprocess
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

epochs = 200

model = Sequential()

df = pd.read_csv('diabetes.csv')

df = preprocess(df)

X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, 'Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

model.add(Dense(32, activation='relu', input_dim=8))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

input(f'Press any key to generate {epochs} epochs. >> ')

model.fit(X_train, y_train, epochs=epochs)

scores = model.evaluate(X_train, y_train)
print('\nTraining Accuracy: %.2f%%\n' % (scores[1] * 100))
scores = model.evaluate(X_test, y_test)
print('Testing Accuracy: %.2f%%\n' % (scores[1] * 100))

see_results = input('Do you want to see the visualized results? [Y/N] >> ')

if (see_results == 'y'):
    y_test_pred = model.predict_classes(X_test)
    c_matrix = confusion_matrix(y_test, y_test_pred)
    ax = sns.heatmap(c_matrix, annot=True, xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'], cbar=False, cmap='Blues')
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    plt.show()
    plt.clf()

    y_test_pred_probs = model.predict(X_test)
    FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
    plt.plot(FPR, TPR)
    plt.plot([0,1],[0,1],'--', color='black')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.clf()

print('Diabetes predictions finished!')