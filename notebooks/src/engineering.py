import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DROP_LATER = ["PassengerId", "Name", "Ticket", "Cabin"]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "SibSp" in df and "Parch" in df:
        df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    return df


class EngineerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=None):
        # IMPORTANT: do not modify params here (clone-safe)
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = engineer_features(pd.DataFrame(X).copy())
        drop_cols = self.drop_cols if self.drop_cols is not None else []
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None
