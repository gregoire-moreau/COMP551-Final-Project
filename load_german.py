

def load():
    X = []
    Y = []
    with open('german.data-numeric', 'r') as f:
        for line in f:
            a = line.split()
            X.append([int(i) for i in a[:-1]])
            Y.append(int(a[-1]))
    return (X, Y)

if __name__ == '__main__':
    (X,Y) = load()
    print(len(X[0]))
    print(len(Y))