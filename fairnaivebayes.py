#https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4909197
import load_german
import load_adult
from sklearn.naive_bayes import MultinomialNB
import numpy as np
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

def CND(X, is_protected, Y, is_desired_outcome):
    (pr, dem) = rank(X, is_protected, Y, is_desired_outcome)
    M = compute_M(X, is_protected, Y, is_desired_outcome)
    print(M)
    for i in range(M):
        Y[pr[i][0]], Y[dem[i][0]]  = Y[dem[i][0]], Y[pr[i][0]]
    #print("disc", discrimination_score(X, Y, is_protected, is_desired_outcome))
    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(X, Y)
    return classifier

def rank(X, is_protected, Y, is_desired_outcome):
    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(X, Y)
    proba = classifier.predict_proba(X)
    print(classifier.classes_)
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


def compare(X, Y, is_protected, is_desired_outcome):
    discrimination_fair = 0
    discrimination_unfair = 0
    score_fair = 0
    score_unfair = 0
    training_X = X[:int(len(X)*4/5)]
    training_Y = Y[:int(len(X)*4/5)]
    validation_X = X[int(len(X)*4/5):]
    validation_Y = Y[int(len(Y)*4/5):]
    fair_classifier = CND(training_X.copy(), is_protected, training_Y.copy(), is_desired_outcome)
    biased_classifier = MultinomialNB(alpha=1.0)
    biased_classifier.fit(training_X, training_Y)
    score_fair += fair_classifier.score(validation_X, validation_Y)
    discrimination_fair += discrimination_score(validation_X, fair_classifier.predict(validation_X), is_protected, is_desired_outcome)
    score_unfair += biased_classifier.score(validation_X, validation_Y)
    discrimination_unfair += discrimination_score(validation_X, biased_classifier.predict(validation_X), is_protected, is_desired_outcome)

    training_X = np.concatenate((X[:int(len(X) * 3 / 5)] , X[int(len(X) * 4 / 5):]), axis=0)
    training_Y = np.concatenate((Y[:int(len(X) * 3 / 5)] , Y[int(len(X) * 4 / 5):]), axis=0)
    validation_X = X[int(len(X) * 3 / 5):int(len(X) * 4 / 5)]
    validation_Y = Y[int(len(Y) * 3 / 5):int(len(X) * 4 / 5)]
    fair_classifier = CND(training_X.copy(), is_protected, training_Y.copy(), is_desired_outcome)
    biased_classifier = MultinomialNB(alpha=1.0)
    biased_classifier.fit(training_X, training_Y)
    score_fair += fair_classifier.score(validation_X, validation_Y)
    discrimination_fair += discrimination_score(validation_X, fair_classifier.predict(validation_X), is_protected, is_desired_outcome)
    score_unfair += biased_classifier.score(validation_X, validation_Y)
    discrimination_unfair += discrimination_score(validation_X, biased_classifier.predict(validation_X), is_protected, is_desired_outcome)

    training_X = np.concatenate((X[:int(len(X) * 2 / 5)] , X[int(len(X) * 3 / 5):]), axis=0)
    training_Y = np.concatenate((Y[:int(len(X) * 2 / 5)] , Y[int(len(X) * 3 / 5):]), axis=0)
    validation_X = X[int(len(X) * 2 / 5):int(len(X) * 3 / 5)]
    validation_Y = Y[int(len(Y) * 2 / 5):int(len(X) * 3 / 5)]
    fair_classifier = CND(training_X.copy(), is_protected, training_Y.copy(), is_desired_outcome)
    biased_classifier = MultinomialNB(alpha=1.0)
    biased_classifier.fit(training_X, training_Y)
    score_fair += fair_classifier.score(validation_X, validation_Y)
    discrimination_fair += discrimination_score(validation_X, fair_classifier.predict(validation_X), is_protected, is_desired_outcome)
    score_unfair += biased_classifier.score(validation_X, validation_Y)
    discrimination_unfair += discrimination_score(validation_X, biased_classifier.predict(validation_X), is_protected, is_desired_outcome)

    training_X = np.concatenate((X[:int(len(X) * 1 / 5)] , X[int(len(X) * 2 / 5):]), axis=0)
    training_Y = np.concatenate((Y[:int(len(X) * 1 / 5)] , Y[int(len(X) * 2 / 5):]), axis=0)
    validation_X = X[int(len(X) * 1 / 5):int(len(X) * 2 / 5)]
    validation_Y = Y[int(len(Y) * 1 / 5):int(len(X) * 2 / 5)]
    fair_classifier = CND(training_X.copy(), is_protected, training_Y.copy(), is_desired_outcome)
    biased_classifier = MultinomialNB(alpha=1.0)
    biased_classifier.fit(training_X, training_Y)
    score_fair += fair_classifier.score(validation_X, validation_Y)
    discrimination_fair += discrimination_score(validation_X, fair_classifier.predict(validation_X), is_protected, is_desired_outcome)
    score_unfair += biased_classifier.score(validation_X, validation_Y)
    discrimination_unfair += discrimination_score(validation_X, biased_classifier.predict(validation_X), is_protected, is_desired_outcome)

    training_X = X[int(len(X) * 1 / 5):]
    training_Y = Y[int(len(Y) * 1 / 5):]
    validation_X = X[:int(len(X) * 1 / 5)]
    validation_Y = Y[:int(len(X) * 1 / 5)]
    fair_classifier = CND(training_X.copy(), is_protected, training_Y.copy(), is_desired_outcome)
    biased_classifier = MultinomialNB(alpha=1.0)
    biased_classifier.fit(training_X, training_Y)
    score_fair += fair_classifier.score(validation_X, validation_Y)
    discrimination_fair += discrimination_score(validation_X, fair_classifier.predict(validation_X), is_protected, is_desired_outcome)
    score_unfair += biased_classifier.score(validation_X, validation_Y)
    discrimination_unfair += discrimination_score(validation_X, biased_classifier.predict(validation_X), is_protected, is_desired_outcome)

    print("Fair classifier : Accuracy = ", score_fair/5
          ,
          " , Discrimination =", discrimination_fair/5)
    print("Biased classifier : Accuracy = ", score_unfair/5
          ,
          " , Discrimination =", discrimination_unfair/5
          )

def german_desired_outcome(val):
    return True if (val == 1) else False

def adult_protected(X):
    return False if X[65] == 1.0 else True

def adult_desired_outcome(val):
    return True if val == ' >50K' else False

if __name__ == '__main__':
    (X, Y) = load_german.load()
    # print(discrimination_score(X, Y, german_protected))
    compare(X, Y, german_protected, german_desired_outcome)
    (X2, Y2) = load_adult.load()
    # print(discrimination_score(X2, Y2, adult_protected, adult_desired_outcome))
    compare(X2, Y2, adult_protected, adult_desired_outcome)
