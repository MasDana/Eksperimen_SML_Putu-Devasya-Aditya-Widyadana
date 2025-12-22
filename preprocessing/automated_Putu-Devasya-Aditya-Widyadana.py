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

    df = df.dropna()
    df = df.drop_duplicates()
    X = df.drop("quality", axis=1)
    y = df["quality"]

    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1

    condition = ~(
        (X < (Q1 - 1.5 * IQR)) |
        (X > (Q3 + 1.5 * IQR))
    ).any(axis=1)

    X = X.loc[condition]
    y = y.loc[condition]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed["quality"] = y.values

    return df_processed


def save_data(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)


def main():

    input_path = "C:/Users/Microsoft/Documents/Eksperimen_SML_Putu_Devasya_Aditya_Widyadana/winequality-white_raw/winequality-white_raw.csv"
    output_path = "C:/Users/Microsoft/Documents/Eksperimen_SML_Putu_Devasya_Aditya_Widyadana/preprocessing/winequality-white_preprocessing/winequality-white_preprocessing.csv"

    df = load_data(input_path)
    df_processed = preprocess_data(df)
    save_data(df_processed, output_path)

    print("Preprocessing completed successfully.")
    print(f"Processed dataset saved at: {output_path}")
    print("Final dataset shape:", df_processed.shape)


if __name__ == "__main__":
    main()
