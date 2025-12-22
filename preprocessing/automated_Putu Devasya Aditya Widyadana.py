import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(input_path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data preprocessing:
    - Drop missing values
    - Drop duplicate rows
    - Remove outliers using IQR method
    - Standardize numerical features
    """
    # Drop missing values
    df = df.dropna()

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Outlier removal using IQR
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1

    condition = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    X = X.loc[condition]
    y = y.loc[condition]

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Combine processed features and target
    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed["target"] = y.values

    return df_processed


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed dataset to CSV.
    """
    df.to_csv(output_path, index=False)


def main():
    input_path = "../heart_disease_raw/heart.csv"
    output_path = "../preprocessing/heart_disease_preprocessing/heart_processed.csv"

    df = load_data(input_path)
    df_processed = preprocess_data(df)
    save_data(df_processed, output_path)

    print("Preprocessing completed. File saved at:", output_path)


if __name__ == "__main__":
    main()
