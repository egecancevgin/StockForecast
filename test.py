import pandas as pd


def test():
    """ Driver test function """
    df = pd.read_csv("PJME_hourly.csv", nrows=39999)
    print(df.head())
    duplicated_data = df[df.duplicated()]
    print("Duplicated Rows:")
    print(duplicated_data)
    nan_data = df[df.isna().any(axis=1)]
    print("\nRows with NaN values:")
    print(nan_data)
    df_cleaned = df.dropna()
    df_cleaned = df_cleaned.drop_duplicates()
    df_cleaned = df_cleaned.sort_values(by='Datetime')
    df_cleaned.to_csv("cln_PJME_hourly.csv", index=False)
    print("Cleaned dataframe exported to cleaned_PJME_hourly.csv")


def main():
    df = pd.read_csv("PJME_hourly.csv")
    df_cleaned = df.dropna()
    df_cleaned = df_cleaned.drop_duplicates()
    df_cleaned = df_cleaned.sort_values(by='Datetime')
    df_cleaned.to_csv("sPJME_hourly.csv", index=False)


def read():
    df = pd.read_csv("sPJME_hourly.csv", nrows=43813)
    df.to_csv("cutPJME_hourly.csv", index=False)


if __name__ == '__main__':
    df = pd.read_csv("PYPL_data.csv")
    df_cleaned = df.sort_values(by='Datetime')
    df.to_csv("paypal.csv", index=False)

