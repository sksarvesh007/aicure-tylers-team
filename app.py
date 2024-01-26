import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error, r2_score

def load_csv(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, target_column=None):
    df = df.drop(['uuid', 'datasetId', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD', 'MEDIAN_REL_RR', 'MEAN_REL_RR',
                'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR', 'KURT_REL_RR', 'SKEW_REL_RR', 'HF_LF'], axis=1)
    df = pd.get_dummies(df, columns=["condition"], drop_first=True)
    
    return df

def train_neural_network(X_train_scaled, y_train):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_scaled, y_train, epochs=80, batch_size=32, validation_split=0.2, verbose=1)
    return model

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, r2

def main():
    data = load_csv('train_data.csv')
    df_nn = preprocess_data(data, target_column='HR')
    X_nn = df_nn.drop(['HR'], axis=1)
    y_nn = df_nn['HR']
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn, y_nn, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled_nn = scaler.fit_transform(X_train_nn)
    X_test_scaled_nn = scaler.transform(X_test_nn)
    nn_model = train_neural_network(X_train_scaled_nn, y_train_nn)
    mae, r2 = evaluate_model(nn_model, X_test_scaled_nn, y_test_nn)
    print(f"Mean Absolute Error on Test Set: {mae}")
    print(f"R-squared on Test Set: {r2}")
    y_pred_test_data = nn_model.predict(X_test_scaled_nn)
    results_test_data_df = pd.DataFrame({'uuid': X_test_nn.index, 'HR': y_pred_test_data.flatten()})
    results_test_data_df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
