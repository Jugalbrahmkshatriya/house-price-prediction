# import pandas as pd
# from sklearn.linear_model import LinearRegression
# def main():
#     data = pd.read_csv("data/house_data.csv")
#     X = data[["area", "bedrooms", "age"]]
#     y = data["price"]
#     model = LinearRegression()
#     model.fit(X, y)
#     sample = [[3000, 3, 15]]
#     prediction = model.predict(sample)
#     print("Predicted price:", prediction[0])
# if __name__ == "__main__":
#     main()





import pandas as pd
from data_loader import load_data, split_features_target
from model import train_model, evaluate_model

# DATA_PATH = "data/house_data.csv"
DATA_PATH = "data/house_data_large.csv"

TARGET_COLUMN = "price"

def main():
    # Step 1: Load data
    data = load_data(DATA_PATH)

    # Step 2: Split features & target
    X, y = split_features_target(data, TARGET_COLUMN)

    # Step 3: Train model
    model, X_test, y_test = train_model(X, y)

    # Step 4: Evaluate model
    mae, r2 = evaluate_model(model, X_test, y_test)

    print("Model evaluation")
    print("MAE:", mae)
    print("R2 Score:", r2)

    # Step 5: Sample prediction
    sample = pd.DataFrame(
        [[3000, 3, 15]],
        columns=X.columns
    )

    prediction = model.predict(sample)
    print("Predicted price:", prediction[0])

if __name__ == "__main__":
    main()
