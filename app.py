import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, columns_to_drop):
    df = df.drop(columns=[col for col in columns_to_drop if col != 'uuid'])
    df = pd.get_dummies(df, columns=["condition"], drop_first=True)
    return df


def generate_report(model, X_test, y_test, df):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    with open("report.txt", "w") as f:
        f.write(f"Number of instances: {len(df)}\n")
        f.write(f"Number of features: {len(df.columns)}\n")
        f.write(f"R-squared score: {r2}\n")
        f.write(f"Mean Absolute Error (MAE): {mae}\n")
        f.write(f"Mean Squared Error (MSE): {mse}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
def train_model(df, target_column):
    X = df.drop(columns=[target_column, 'uuid'])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    generate_report(model, X_test, y_test , df)

    return model

def main():
    file_path = "train_data.csv"
    df = load_csv(file_path)
    columns_to_drop = ["datasetId", "SDRR" , "RMSSD" , "SDSD" , "SDRR_RMSSD" , "MEDIAN_REL_RR" , "SDRR_REL_RR" , "RMSSD_REL_RR" , "SDSD_REL_RR" , "SDRR_RMSSD_REL_RR" , "KURT_REL_RR" , "SKEW_REL_RR" , "HF_LF"]
    df = preprocess_data(df, columns_to_drop)
    target_column = "HR"
    model = train_model(df, target_column)
    predictions = model.predict(df.drop(columns=[target_column, 'uuid']))
    predictions_df = pd.DataFrame({'uuid': df['uuid'], 'HR': predictions})

    predictions_df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()