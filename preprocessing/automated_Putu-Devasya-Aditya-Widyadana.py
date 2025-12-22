"""
Automated Data Preprocessing Script
Wine Quality Dataset

"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(input_path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    # 1. Handle missing values
    df = df.dropna()

    # 2. Remove duplicate rows
    df = df.drop_duplicates()

    # 3. Separate features and target
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # 4. Outlier detection & removal using IQR
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1

    condition = ~(
        (X < (Q1 - 1.5 * IQR)) |
        (X > (Q3 + 1.5 * IQR))
    ).any(axis=1)

    X = X.loc[condition]
    y = y.loc[condition]

    # 5. Feature scaling (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Combine processed features and target
    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed["quality"] = y.values

    return df_processed


def save_data(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)


def main():
    # Path configuration
    input_path = "C:/Users/Microsoft/Documents/Eksperimen_SML_Putu_Devasya_Aditya_Widyadana/winequality-white_raw/winequality-white_raw.csv"
    output_path = "C:/Users/Microsoft/Documents/Eksperimen_SML_Putu_Devasya_Aditya_Widyadana/preprocessing/winequality-white_preprocessing/winequality-white_preprocessing.csv"

    # Run preprocessing pipeline
    df = load_data(input_path)
    df_processed = preprocess_data(df)
    save_data(df_processed, output_path)

    print("Preprocessing completed successfully.")
    print(f"Processed dataset saved at: {output_path}")
    print("Final dataset shape:", df_processed.shape)


if __name__ == "__main__":
    main()
