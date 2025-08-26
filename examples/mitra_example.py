import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

# Load datasets
wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

print("Dataset shapes:")
print(f"Wine: {wine_df.shape}")

# Create train/test splits (80/20)
wine_train, wine_test = train_test_split(
    wine_df, test_size=0.2, random_state=42, stratify=wine_df['target'])

print("Training set sizes:")
print(f"Wine: {len(wine_train)} samples")

# Convert to TabularDataset
wine_train_data = TabularDataset(wine_train)
wine_test_data = TabularDataset(wine_test)

# Create predictor with Mitra
print("Training Mitra classifier on classification dataset...")
mitra_predictor = TabularPredictor(label='target')
mitra_predictor.fit(
    wine_train_data,
    hyperparameters={
        'MITRA': {'fine_tune': False}
    },
)

print("\nMitra training completed!")

# Make predictions
mitra_predictions = mitra_predictor.predict(wine_test_data)
print("Sample Mitra predictions:")
print(mitra_predictions.head(10))

# Show prediction probabilities for first few samples
mitra_predictions = mitra_predictor.predict_proba(wine_test_data)
print(mitra_predictions.head())

# Show model leaderboard
print("\nMitra Model Leaderboard:")
mitra_predictor.leaderboard(wine_test_data)
