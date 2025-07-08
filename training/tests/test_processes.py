import pytest
from data_processing.process import DataPreprocessor, PreprocessingConfig

@pytest.fixture(scope="module")
def config():
    return PreprocessingConfig(
        file_path="data/DataScientist_CaseStudy_Dataset.xlsx",
    )

@pytest.fixture(scope="module")
def preprocessor(config):
    dp = DataPreprocessor(config)
    dp.load_and_merge()
    return dp

def test_preprocess_output(preprocessor):
    df = preprocessor.preprocess()
    assert df.isnull().sum().sum() == 0
    assert df.shape[0] > 0
    assert "Tenure" in df.columns
    assert any(col.startswith("Sex_") for col in df.columns)

def test_split_shapes(preprocessor):
    preprocessor.preprocess()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
    total_rows = len(preprocessor.df)
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == total_rows
    assert y_train.shape[0] == X_train.shape[0]