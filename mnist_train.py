from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
print("If this is your first time running the program, please make sure you are connected to the internet.")

try:
    print('It will download automaticaly if you haven\'t do so(it will print out some strange things).\nIt may take some time.')
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    sc1 = accuracy_score(y_test, pred)
    print("Accuracy of Logistic Regression:", sc1)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    sc2 = accuracy_score(y_test, pred)
    print("Accuracy of KNN:", sc2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    sc3 = accuracy_score(y_test, pred)
    print("Accuracy of Random Forest:", sc3)

    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train_pca, y_train)
    pred = model.predict(X_test_pca)
    sc4 = accuracy_score(y_test, pred)
    print("Accuracy of PCA + Logistic Regression:", sc4)

    print('Calculating total results...')
    time.sleep(3)

    results = {"Logistic Regression": sc1, "KNN": sc2, "Random Forest": sc3, "PCA + Logistic Regression": sc4}

    ranking = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f'First: {ranking[0]}')
    print(f'Second: {ranking[1]}')
    print(f'Third: {ranking[2]}')
    print(f'Fourth: {ranking[3]}')
except Exception as e:
    print('Failed to download MNIST...')
    print(f'Error details: {e}')
