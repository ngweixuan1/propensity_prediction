from typing import List, Tuple, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class PreprocessingConfig(BaseModel):
    file_path: str = Field(..., description="Path to Excel file")
    label_cols: List[str] = Field(
        default=['Sale_MF', 'Sale_CC', 'Sale_CL', 'Revenue_MF', 'Revenue_CC', 'Revenue_CL']
    )
    exclude_na_col: str = Field(default='Sex')
    test_size: float = Field(default=0.1)
    val_size: float = Field(default=0.1)
    random_state: Optional[int] = 42
    sheet_map: dict = Field(default_factory=lambda: {
        'soc_dem': 'Soc_Dem',
        'products': 'Products_ActBalance',
        'inflow_outflow': 'Inflow_Outflow',
        'sales_revenue': 'Sales_Revenues'
    })


class DataPreprocessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.df = None

    def load_and_merge(self) -> pd.DataFrame:
        sheets = {
            key: pd.read_excel(self.config.file_path, sheet_name=sheet)
            for key, sheet in self.config.sheet_map.items()
        }

        print({key: f"{df.shape}" for key, df in sheets.items()})

        df = sheets['sales_revenue'] \
            .merge(sheets['products'], on='Client', how='left') \
            .merge(sheets['inflow_outflow'], on='Client', how='left') \
            .merge(sheets['soc_dem'], on='Client', how='left')

        print("Merged shape:", df.shape)
        print("Duplicated Clients:", df.duplicated(subset="Client").sum())

        self.df = df
        return df

    def preprocess(self) -> pd.DataFrame:
        df = self.df.copy()
        label_cols = self.config.label_cols
        feature_cols = [col for col in df.columns if col not in label_cols + ['Client']]

        # Fill missing except excluded column
        exclude_col = self.config.exclude_na_col
        df[df.columns.difference([exclude_col])] = df[df.columns.difference([exclude_col])].fillna(0)
        df = df.dropna()
        print("After dropping NA:", df.shape)

        # Filter erroneous tenure
        df = df[df['Tenure'] / 12 <= df['Age']]
        print("After tenure filtering:", df.shape)

        # One-hot encoding
        if exclude_col in df.columns:
            dummies = pd.get_dummies(df[exclude_col], prefix=exclude_col, drop_first=True).astype(int)
            df = df.drop(exclude_col, axis=1).join(dummies)

        # Log transform certain columns
        to_log = [col for col in df.columns if col.startswith('Volume') or col.startswith('Transactions')]
        df[to_log] = df[to_log].apply(lambda x: np.log1p(x))

        self.df = df
        return df

    def split_data(self) -> Tuple[pd.DataFrame, ...]:
        df = self.df.copy()
        label_cols = [col for col in self.config.label_cols if col.startswith("Sale_")]
        stratify_key = df[label_cols].astype(str).agg('_'.join, axis=1)

        X = df.drop(columns=self.config.label_cols)
        y = df[self.config.label_cols]

        X_train_val, X_test, y_train_val, y_test, strat_train_val, _ = train_test_split(
            X, y, stratify_key, test_size=self.config.test_size,
            random_state=self.config.random_state, shuffle=True
        )

        val_adjusted = self.config.val_size / (1 - self.config.test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_adjusted,
            stratify=strat_train_val,
            random_state=self.config.random_state, shuffle=True
        )

        print("Split shapes:")
        print("Train:", X_train.shape)
        print("Val:", X_val.shape)
        print("Test:", X_test.shape)

        return X_train, X_val, X_test, y_train, y_val, y_test