import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_csv(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, target_column):
    df = pd.get_dummies(df, columns=["condition"], drop_first=True)
    return df

def generate_report(model, X_test, y_test, df):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    f_score = model.score(X_test, y_test)

    with open("report.txt", "w") as f:
        f.write(f"Number of instances: {len(df)}\n")
        f.write(f"Number of features: {len(df.columns)}\n")
        f.write(f"R-squared score: {r2}\n")
        f.write(f"Mean Absolute Error (MAE): {mae}\n")
        f.write(f"Mean Squared Error (MSE): {mse}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
        f.write(f"F-score: {f_score}\n")

def main():
    file_path = "train_data.csv"
    df = load_csv(file_path)
    target_column = "HR"
    df = preprocess_data(df, target_column)
    model = RandomForestRegressor()
    model.fit(df.drop(columns=[target_column, 'uuid']), df[target_column])
    feature_importance = model.feature_importances_
    top_features = df.drop(columns=[target_column, 'uuid']).columns[np.argsort(feature_importance)[::-1]][:15]
    X_train, X_test, y_train, y_test = train_test_split(df[top_features], df[target_column], test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, df[top_features], df[target_column], cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    print("Cross-Validated RMSE:", cv_rmse)
    print("Cross-Validated r2:", cross_val_score(model, df[top_features], df[target_column], cv=5, scoring='r2').mean())
    generate_report(model, X_test, y_test, df[top_features])
    predictions = model.predict(df[top_features])
    predictions_df = pd.DataFrame({'uuid': df['uuid'], 'HR': predictions})
    predictions_df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
