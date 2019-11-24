import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def test_embedding_on_ideology(dataset):
    X = dataset[dataset.columns.difference(['mapped_id', 'theta', "country","raw_location", "latitude", "longitude"])]
    y = dataset[["theta"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    print("testing embedding's predictiveness on ideology...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("R^2 score: " + str(score))
    print("predictions:")
    print(model.predict(X_test))
    print("Y:")
    print(y_test)
    
def test_ideology_geo(dataset):
    X = dataset[dataset.columns.difference(['mapped_id', 'theta', "country","raw_location", "latitude", "longitude"])]
    y = dataset[["theta", "latitude", "longitude"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    tree_model = RandomForestRegressor()
    tree_model.fit(X_train, y_train)
    score = tree_model.score(X_test, y_test)
    print("R^2 score: " + str(score))
    print("predictions:")
    print(model.predict(X_test))
    print("Y:")
    print(y_test)

def run_model():
    transformed_embedding_file = "embedding_dataset.csv"
    dataset = pd.read_csv(transformed_embedding_file)
    test_embedding_on_ideology(dataset)
    test_ideology_geo(dataset)

if __name__ == "__main__":
    run_model()