#steps/train.py
from src.data_loader import DataLoader
from src.feature_extraction import FeatureEngineer
from src.label_creator import LabelCreator
from src.model import LiquidityStressModel
from config.settings import PARQUET_OUTPUT_PATH, MODEL_PATH


if __name__ == "__main__":
   file = PARQUET_OUTPUT_PATH

   feature_cols = [
       'spread_bps', 'spread_ma_5m', 'spread_std_5m',
       'volume_ma_5m', 'imbalance_ma_5m',
       'price_change_1m', 'price_change_5m', 'volatility_5m',
       'spread_trend', 'volume_surge'
   ]

   loader = DataLoader(file)
   df = loader.load_data()

   engineer = FeatureEngineer()
   df = engineer.create_features(df)

   labeler = LabelCreator(horizon_minutes=10)
   df = labeler.create_labels(df)

   label_counts = df['stress_label'].value_counts()
   total = len(df)
   print("\nLabel distribution:")
   print(label_counts)
   model = LiquidityStressModel()
   X, y = model.prepare_data(df, feature_cols)

   scores = model.walk_forward_validation(X, y, n_splits=5)

   model.train_final_model(X, y)

   latest_features = X[-1]
   prediction = model.predict_stress(latest_features)

   model.save_model(MODEL_PATH)

