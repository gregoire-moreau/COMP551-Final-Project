import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB

def load():
    X = []
    Y = []
    with open('german.data-numeric', 'r') as f:
        for line in f:
            a = line.split()
            X.append([int(i) for i in a[:-1]])
            Y.append(int(a[-1]))
    #X = StandardScaler(with_mean=False).fit_transform(X)
    return (X, Y)

if __name__ == '__main__':
    (X,Y) = load()
    print(X[10])
    #print(len(Y))