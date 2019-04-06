#https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4909197
import load_german
import load_adult
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
#German dataset


def discrimination_score(X, Y, is_protected, is_desired_outcome):
    aged = 0
    young = 0
    young_pos = 0
    aged_pos = 0
    for i in range(len(X)):
        if is_protected(X[i]):
            young += 1
            if is_desired_outcome(Y[i]):
                young_pos += 1
        else:
            aged += 1
            if is_desired_outcome(Y[i]):
                aged_pos +=1
    # print(aged, young)
    return ((aged_pos)/aged) - ((young_pos)/young)

def compute_M(X, is_protected, Y, is_desired_outcome):
    aged = 0
    young = 0
    young_pos = 0
    aged_pos = 0
    for i in range(len(X)):
        if is_protected(X[i]):
            young += 1
            if is_desired_outcome(Y[i]):
                young_pos += 1
        else:
            aged += 1
            if is_desired_outcome(Y[i]):
                aged_pos += 1
    return int(((young*aged_pos)-(aged*young_pos))/(young+aged))

def german_protected(X):
    return True if X[9] <= 25 else False
def german_desired_outcome(val):
    return True if (val == 1) else False

def adult_protected(X):
    return False if X[65] == 1.0 else True

def adult_desired_outcome(val):
    return True if val == ' >50K' else False

class CNDClassifier(BaseEstimator):
    def __init__(self, is_protected= adult_protected, is_desired_outcome=adult_desired_outcome, goal='disc', alpha = 1e-10, max_depth = None):
        self.alpha = alpha
        self.classifier = None
        self.is_protected = is_protected
        self.is_desired_outcome = is_desired_outcome
        self.goal = goal
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.classifier = CND(X, self.is_protected, Y, self.is_desired_outcome, self.alpha, self.max_depth)
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def get_params(self, deep=True):
        return {'alpha':self.alpha, 'max_depth':self.max_depth}

    def set_params(self, **params):
        self.alpha = params['alpha']
        self.max_depth = params['max_depth']
        return self

    def score(self, X, Y):
        acc = self.classifier.score(X, Y)
        if self.goal == 'delta':
            disc = discrimination_score(X, self.classifier.predict(X), self.is_protected, self.is_desired_outcome)
            return acc- disc
        elif self.goal == 'accuracy':
            return acc
        else:
            disc = discrimination_score(X, self.classifier.predict(X), self.is_protected, self.is_desired_outcome)
            return -disc


def CND(X, is_protected, Y, is_desired_outcome, alpha= 1e-10, max_depth = None):
    (pr, dem) = rank(X, is_protected, Y, is_desired_outcome, alpha)
    M = compute_M(X, is_protected, Y, is_desired_outcome)
    #print(M)
    for i in range(M):
        Y[pr[i][0]], Y[dem[i][0]]  = Y[dem[i][0]], Y[pr[i][0]]
    #print("disc", discrimination_score(X, Y, is_protected, is_desired_outcome))
    #classifier = MultinomialNB(alpha=alpha)
    if (max_depth != None):
        classifier = DecisionTreeClassifier(class_weight='balanced', max_depth=max_depth)
    else:
        classifier = MultinomialNB(alpha=alpha)
    classifier.fit(X, Y)
    return classifier

def rank(X, is_protected, Y, is_desired_outcome, alpha=1e-10, max_depth = None ):
    #classifier = MultinomialNB(alpha=alpha)
    if (max_depth != None):
        classifier = DecisionTreeClassifier(class_weight='balanced', max_depth=max_depth)
    else:
        classifier = MultinomialNB(alpha=alpha)

    classifier.fit(X, Y)
    proba = classifier.predict_proba(X)
    #print(classifier.classes_)
    pr = []
    dem = []
    index = -1
    for i in range(len(classifier.classes_)):
        if is_desired_outcome(classifier.classes_[i]):
            index = i
            break
    for i in range(len(X)):
        if is_protected(X[i]) and not is_desired_outcome(Y[i]):
            pr.append([i, proba[i][index]])
        elif not is_protected(X[i]) and is_desired_outcome(Y[i]):
            dem.append([i, proba[i][index]])
    return (sorted(pr, key=lambda X:X[1], reverse = True), sorted(dem, key=lambda X:X[1], reverse = False))


def compare(X, Y, is_protected, is_desired_outcome, alpha, max_depth = None):
    discrimination_fair = 0
    discrimination_unfair = 0
    score_fair = 0
    score_unfair = 0
    consistency_fair = 0
    consistency_unfair = 0
    training_X = X[:int(len(X)*4/5)]
    training_Y = Y[:int(len(X)*4/5)]
    validation_X = X[int(len(X)*4/5):]
    validation_Y = Y[int(len(Y)*4/5):]
    fair_classifier = CND(training_X.copy(), is_protected, training_Y.copy(), is_desired_outcome, alpha, max_depth)
    biased_classifier = MultinomialNB(alpha=alpha)
    biased_classifier.fit(training_X, training_Y)
    score_fair += fair_classifier.score(validation_X, validation_Y)
    discrimination_fair += discrimination_score(validation_X, fair_classifier.predict(validation_X), is_protected, is_desired_outcome)
    score_unfair += biased_classifier.score(validation_X, validation_Y)
    discrimination_unfair += discrimination_score(validation_X, biased_classifier.predict(validation_X), is_protected, is_desired_outcome)
    consistency_fair += consistency(validation_X, fair_classifier.predict(validation_X), 5)
    consistency_unfair += consistency(validation_X, biased_classifier.predict(validation_X), 5)

    training_X = np.concatenate((X[:int(len(X) * 3 / 5)] , X[int(len(X) * 4 / 5):]), axis=0)
    training_Y = np.concatenate((Y[:int(len(X) * 3 / 5)] , Y[int(len(X) * 4 / 5):]), axis=0)
    validation_X = X[int(len(X) * 3 / 5):int(len(X) * 4 / 5)]
    validation_Y = Y[int(len(Y) * 3 / 5):int(len(X) * 4 / 5)]
    fair_classifier = CND(training_X.copy(), is_protected, training_Y.copy(), is_desired_outcome, alpha, max_depth)
    biased_classifier = MultinomialNB(alpha=alpha)
    biased_classifier.fit(training_X, training_Y)
    score_fair += fair_classifier.score(validation_X, validation_Y)
    discrimination_fair += discrimination_score(validation_X, fair_classifier.predict(validation_X), is_protected, is_desired_outcome)
    score_unfair += biased_classifier.score(validation_X, validation_Y)
    discrimination_unfair += discrimination_score(validation_X, biased_classifier.predict(validation_X), is_protected, is_desired_outcome)
    consistency_fair += consistency(validation_X, fair_classifier.predict(validation_X), 5)
    consistency_unfair += consistency(validation_X, biased_classifier.predict(validation_X), 5)

    training_X = np.concatenate((X[:int(len(X) * 2 / 5)] , X[int(len(X) * 3 / 5):]), axis=0)
    training_Y = np.concatenate((Y[:int(len(X) * 2 / 5)] , Y[int(len(X) * 3 / 5):]), axis=0)
    validation_X = X[int(len(X) * 2 / 5):int(len(X) * 3 / 5)]
    validation_Y = Y[int(len(Y) * 2 / 5):int(len(X) * 3 / 5)]
    fair_classifier = CND(training_X.copy(), is_protected, training_Y.copy(), is_desired_outcome, alpha, max_depth)
    biased_classifier = MultinomialNB(alpha=alpha)
    biased_classifier.fit(training_X, training_Y)
    score_fair += fair_classifier.score(validation_X, validation_Y)
    discrimination_fair += discrimination_score(validation_X, fair_classifier.predict(validation_X), is_protected, is_desired_outcome)
    score_unfair += biased_classifier.score(validation_X, validation_Y)
    discrimination_unfair += discrimination_score(validation_X, biased_classifier.predict(validation_X), is_protected, is_desired_outcome)
    consistency_fair += consistency(validation_X, fair_classifier.predict(validation_X), 5)
    consistency_unfair += consistency(validation_X, biased_classifier.predict(validation_X), 5)

    training_X = np.concatenate((X[:int(len(X) * 1 / 5)] , X[int(len(X) * 2 / 5):]), axis=0)
    training_Y = np.concatenate((Y[:int(len(X) * 1 / 5)] , Y[int(len(X) * 2 / 5):]), axis=0)
    validation_X = X[int(len(X) * 1 / 5):int(len(X) * 2 / 5)]
    validation_Y = Y[int(len(Y) * 1 / 5):int(len(X) * 2 / 5)]
    fair_classifier = CND(training_X.copy(), is_protected, training_Y.copy(), is_desired_outcome, alpha, max_depth)
    biased_classifier = MultinomialNB(alpha=alpha)
    biased_classifier.fit(training_X, training_Y)
    score_fair += fair_classifier.score(validation_X, validation_Y)
    discrimination_fair += discrimination_score(validation_X, fair_classifier.predict(validation_X), is_protected, is_desired_outcome)
    score_unfair += biased_classifier.score(validation_X, validation_Y)
    discrimination_unfair += discrimination_score(validation_X, biased_classifier.predict(validation_X), is_protected, is_desired_outcome)
    consistency_fair += consistency(validation_X, fair_classifier.predict(validation_X), 5)
    consistency_unfair += consistency(validation_X, biased_classifier.predict(validation_X), 5)

    training_X = X[int(len(X) * 1 / 5):]
    training_Y = Y[int(len(Y) * 1 / 5):]
    validation_X = X[:int(len(X) * 1 / 5)]
    validation_Y = Y[:int(len(X) * 1 / 5)]
    fair_classifier = CND(training_X.copy(), is_protected, training_Y.copy(), is_desired_outcome, alpha, max_depth)
    biased_classifier = MultinomialNB(alpha=alpha)
    biased_classifier.fit(training_X, training_Y)
    score_fair += fair_classifier.score(validation_X, validation_Y)
    discrimination_fair += discrimination_score(validation_X, fair_classifier.predict(validation_X), is_protected, is_desired_outcome)
    score_unfair += biased_classifier.score(validation_X, validation_Y)
    discrimination_unfair += discrimination_score(validation_X, biased_classifier.predict(validation_X), is_protected, is_desired_outcome)
    consistency_fair += consistency(validation_X, fair_classifier.predict(validation_X), 5)
    consistency_unfair += consistency(validation_X, biased_classifier.predict(validation_X), 5)

    print("Fair classifier : Accuracy = ", score_fair/5
          ,
          " , Discrimination =", discrimination_fair/5,
          "Consistency = ", consistency_fair/5)
    print("Biased classifier : Accuracy = ", score_unfair/5
          ,
          " , Discrimination =", discrimination_unfair/5,
          "Consistency = ", consistency_unfair/5
          )



def consistency(X,Y, k):
    mod = NearestNeighbors(n_neighbors=(k+1))
    mod.fit(X)
    sum = 0
    for i in range(len(X)):
        neighbors = mod.kneighbors([X[i]])
        for j in neighbors[1][0]:
            if j != i and (Y[i] != Y[j]):
                sum += 1
    return 1-sum/(len(X)*k)

if __name__ == '__main__':

    print("Naive Bayes")
    print("German dataset")

    (X, Y) = load_german.load()

    compare(X, Y, german_protected, german_desired_outcome, 1e-10)
    print("Adult dataset")
    (X2, Y2) = load_adult.load()
    compare(X2, Y2, adult_protected, adult_desired_outcome, 1e-10)
    print("Decision trees")
    print("German dataset")
    (X, Y) = load_german.load()

    compare(X, Y, german_protected, german_desired_outcome, 1e-10, max_depth = 20)
    print("Adult dataset")
    (X2, Y2) = load_adult.load()
    compare(X2, Y2, adult_protected, adult_desired_outcome, 1e-10, max_depth=44)

    '''
    params_to_test = [{'alpha' : [1e-10] , 'max_depth':[i for i in range(1, 20)]}]
    gs = GridSearchCV(CNDClassifier(), params_to_test)
    gs.fit(X2, Y2)
    print(gs.best_params_)
    '''
