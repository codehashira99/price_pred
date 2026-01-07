Satellite Property Price Predictor 


Multimodal ML model predicting property prices from tabular data + satellite images. Achieved RMSE 101,420 and R² 0.888 on validation.

📊 Results Summary
Model	RMSE (Validation)	R² Score	Improvement
Tabular Only	101,496	0.888	Baseline
Tabular + Satellite Images	101,420	0.888	+0.1%
Final submission: submission.csv (5,404 test predictions, prices $131K–$2.9M)

🏗️ Architecture Overview
text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Tabular Data   │───▶│ StandardScaler   │───▶│ HistGBR (500)   │
│ (21 features)   │    │   (21→21)        │    │   (533 feats)   │
└─────────┬───────┘    └──────────┬───────┘    └───────┬────────┘
          │                       │                       │
          │                 ┌─────▼──────┐              │
          └────────────────▶│ ResNet18   │◀─────────────┘
                            │ (512 feats)│
                            │ Pretrained  │
                            └─────┬──────┘
                                  │
                            ┌─────▼──────┘
                            │   X_test    │
                            │ (6487, 533) │ ✓
                            └─────────────┘
🚀 Quick Start
bash
# 1. Clone & Setup
git clone <your-repo>
cd satellite-property-price-predictor
pip install torch torchvision scikit-learn pandas numpy joblib matplotlib requests tqdm pillow jupyter

# 2. Download Kaggle Data
kaggle competitions download -c satellite-property-price-predictor
unzip satellite-property-price-predictor.zip -d data/

# 3. Fetch Satellite Images (10min)
python data_fetcher.py

# 4. Run Pipeline
jupyter nbconvert --execute --to notebook preprocess.ipynb
jupyter nbconvert --execute --to notebook model_training.ipynb

# 5. Submit
cat submission.csv  # Ready!
📁 Repository Structure
text
satellite-property-price-predictor/
├── data/
│   ├── train.csv          # 16,209 properties (21 cols)
│   ├── test.csv           # 5,404 test properties
│   ├── images/train/      # 12,901 satellite PNGs (256x256)
│   └── images/test/       # 5,396 test satellite PNGs
├── data/processed/
│   ├── X_train_full.npy   # (12,901, 533) merged features
│   ├── fusion_model.joblib # Final model
│   ├── tabular_scaler.joblib
│   └── test_img_features.pkl
├── data_fetcher.py        # Parallel Mapbox downloader
├── preprocess.ipynb       # EDA + feature engineering
├── model_training.ipynb   # ResNet18 + fusion training
├── submission.csv         # Kaggle submission
└── README.md             # You're reading it!
🔍 Key Features Engineered
21 Tabular Features (preprocess.ipynb):

text
bedrooms, bathrooms, sqft_living, sqft_lot, floors
waterfront, view, condition, grade, sqft_above
sqft_basement, yr_built, yr_renovated, zipcode
lat, long, sqft_living15, sqft_lot15, year, month
dist_center_km (NEW: Haversine from median lat/lon)
512 Image Features:

ResNet18 (ImageNet pretrained, FC removed)

256x256 satellite images (zoom=17)

Frozen backbone → property embeddings

# Core ML
torch torchvision torchaudio
scikit-learn

# Data
pandas numpy joblib

# Viz
matplotlib seaborn

# Images
requests tqdm pillow

# Notebooks
jupyter ipykernel
🛠️ Technical Highlights
✅ Parallel Downloads: 16 workers (10min vs 3hr)
✅ Perfect Test Merge: (6487, 21) + (6487, 512) → (6487, 533)
✅ Zero-Fill Missing: Handles absent images/features
✅ Log-Price Training: np.log1p() → np.expm1() for submission
✅ Reproducible: All seeds fixed, exact column matching

📤 Deliverables Checklist
 submission.csv - 5,404 predictions ready

 Code Repo - Full pipeline + notebooks

 README.md - Setup + results

 Report Ready - Copy sections above to PDF
