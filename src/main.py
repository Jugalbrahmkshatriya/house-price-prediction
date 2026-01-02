import pandas as pd
from sklearn.linear_model import LinearRegression
def main():
    data = pd.read_csv("data/house_data.csv")
    X = data[["area", "bedrooms", "age"]]
    y = data["price"]
    model = LinearRegression()
    model.fit(X, y)
    sample = [[3000, 3, 15]]
    prediction = model.predict(sample)
    print("Predicted price:", prediction[0])
if __name__ == "__main__":
    main()