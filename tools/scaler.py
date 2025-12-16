import pandas as pd
from sklearn.preprocessing import StandardScaler


def scaler(df, target):
    features = df.drop(columns=target)
    df_target = df[target]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    final_df = pd.concat([features_scaled_df, df_target], axis=1)
    return final_df
