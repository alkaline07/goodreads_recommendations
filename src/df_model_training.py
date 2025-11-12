import pandas as pd

class DataFrameModelTraining:
    def __init__(self, df):
        self.df = df

    def train_model(self):
        print("Training model with DataFrame of shape:", self.df.shape)

if __name__ == "__main__":

    # For demonstration, load a sample DataFrame
    df = pd.read_parquet("data/train_data.parquet")
    
    model_trainer = DataFrameModelTraining(df)
    model_trainer.train_model()