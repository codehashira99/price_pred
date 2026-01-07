# Satellite Property Price Predictor

A multimodal machine learning model that predicts property prices using both tabular data and satellite imagery. This project combines traditional property features with visual information extracted from satellite images using deep learning.

## 📊 Project Overview

This project implements a hybrid approach to property price prediction:
- **Tabular Model**: Uses property features (bedrooms, bathrooms, location, etc.)
- **Image Model**: Extracts features from satellite imagery using ResNet18
- **Fusion Model**: Combines both feature sets for improved predictions

**Performance Metrics:**
- RMSE (Validation): 101,420
- R² Score: 0.888
- Test Predictions: 5,404 properties with prices ranging from $131K to $2.9M

## 📁 Project Structure

```
price_pred/
├── data/
│   ├── train.csv                    # Training data (16,209 properties with 21 features)
│   ├── test.csv                     # Test data (5,404 properties)
│   ├── images/
│   │   ├── train/                   # Training satellite images (256x256 PNG)
│   │   └── test/                    # Test satellite images (256x256 PNG)
│   └── processed/
│       ├── X_train_full.npy         # Processed training features
│       ├── fusion_model.joblib      # Trained fusion model
│       ├── tabular_scaler.joblib    # Feature scaler
│       └── test_img_features.pkl    # Processed test image features
├── models/                           # Saved model checkpoints
├── data_fetcher.py                   # Script to download satellite images from Mapbox API
├── preprocess.ipynb                  # Data exploration and feature engineering
├── model_training.ipynb              # Model training and evaluation
├── submission.csv                    # Final predictions for submission
├── submission_tabular.csv            # Baseline tabular-only predictions
├── Report.pdf                        # Detailed project report
├── .gitignore                        # Git ignore file
└── README.md                         # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Internet connection (for downloading satellite images)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/codehashira99/price_pred.git
   cd price_pred
   ```

2. **Install required packages**
   ```bash
   pip install torch torchvision scikit-learn pandas numpy joblib matplotlib requests tqdm pillow jupyter seaborn
   ```

   Or install from requirements file:
   ```bash
   pip install -r requirements.txt
   ```

### Setup Data

1. **Download the dataset**
   
   If using Kaggle competition data:
   ```bash
   unzip satellite-property-price-predictor.zip -d data/
   ```

   Or manually place your `train.csv` and `test.csv` files in the `data/` directory.

2. **Fetch satellite images**
   
   This script downloads satellite images from Mapbox API for all properties:
   ```bash
   python data_fetcher.py
   ```
   
   **Note:** This may take approximately 10 minutes with parallel downloading. You'll need a Mapbox API token (update the token in `data_fetcher.py` if required).

## 📋 Running the Pipeline

Follow these steps in order to reproduce the results:

### Step 1: Data Preprocessing and Feature Engineering

Open and run the preprocessing notebook:
```bash
jupyter notebook preprocess.ipynb
```

Or execute directly:
```bash
jupyter nbconvert --execute --to notebook preprocess.ipynb
```

This notebook:
- Performs exploratory data analysis (EDA)
- Engineers 21 features from tabular data
- Creates derived features (distance to center, year/month extraction)
- Handles missing values
- Saves processed data

### Step 2: Model Training

Open and run the model training notebook:
```bash
jupyter notebook model_training.ipynb
```

Or execute directly:
```bash
jupyter nbconvert --execute --to notebook model_training.ipynb
```

This notebook:
- Loads preprocessed tabular data
- Extracts image features using ResNet18 (pretrained on ImageNet)
- Merges tabular and image features (533 total features)
- Trains the fusion model
- Evaluates on validation set
- Generates predictions on test set
- Creates `submission.csv` file

### Step 3: Generate Submission

After running both notebooks, your submission file will be ready:
```bash
cat submission.csv
```

The file contains predictions for all test properties in the format required for submission.

## 🔧 Key Features

### Tabular Features (21 total)
- **Property Characteristics**: bedrooms, bathrooms, sqft_living, sqft_lot, floors
- **Quality Indicators**: waterfront, view, condition, grade
- **Structure Details**: sqft_above, sqft_basement, yr_built, yr_renovated
- **Location**: zipcode, lat, long, sqft_living15, sqft_lot15
- **Temporal**: year, month
- **Derived**: dist_center_km (Haversine distance from median location)

### Image Features (512 total)
- Extracted using ResNet18 (ImageNet pretrained)
- 256x256 satellite images at zoom level 17
- Frozen backbone for transfer learning
- Captures visual property characteristics

## 📈 Model Architecture

1. **Tabular Processing**
   - StandardScaler normalization
   - 21 engineered features

2. **Image Processing**
   - ResNet18 (pretrained, FC layer removed)
   - Global Average Pooling
   - 512-dimensional feature vectors

3. **Fusion Model**
   - Combined feature vector (533 dimensions)
   - Ensemble or stacked model approach
   - Log-transformed target (prices)

## 🎯 Results

| Model | RMSE (Validation) | R² Score | Notes |
|-------|-------------------|----------|-------|
| Tabular Only | 101,496 | 0.888 | Baseline |
| Tabular + Satellite Images | 101,420 | 0.888 | +0.1% improvement |

## 🛠️ Technical Highlights

- ✅ **Parallel Downloads**: 16 workers for fast image acquisition
- ✅ **Robust Preprocessing**: Zero-fill for missing images/features
- ✅ **Log-Price Training**: `np.log1p()` → `np.expm1()` for better distribution
- ✅ **Reproducible**: All random seeds fixed
- ✅ **Production Ready**: Complete pipeline from raw data to submission

## 📝 Notes

- The model uses log-transformed prices during training for better performance
- Missing satellite images are handled gracefully with zero-filled features
- All preprocessing steps are saved for consistent test-time transformation
- Random seeds are fixed for reproducibility

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements.

## 📄 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Predicting! 🏠📈**
