# ===================================
# COMPLETE HEART DISEASE DATA PREPROCESSING PIPELINE
# Steps 1-5: Load → Handle Missing → Encode → Scale → EDA
# ===================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("🚀 HEART DISEASE DATA PREPROCESSING PIPELINE")
print("="*80)

# ===================================
# STEP 1: LOAD DATASET
# ===================================
print("\n📥 STEP 1: LOADING HEART DISEASE DATASET FROM UCI")
print("="*60)

try:
    from ucimlrepo import fetch_ucirepo
    
    # Fetch Heart Disease dataset (ID=45)
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Combine features and target
    df = pd.concat([X, y], axis=1)
    
    print("✅ Dataset loaded successfully!")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🎯 Target variable: {y.columns[0]}")
    
    # Display first few rows
    print("\n📋 First 5 rows:")
    print(df.head())
    
 # <<< أضف الكود الجديد هنا
    # Convert target variable 'num' to binary format
    # 0 = No Disease
    # 1 = Disease (if original value was 1, 2, 3, or 4)
    # ==============================================================
    df['num'] = (df['num'] > 0).astype(int)
    print("\n✅ Target variable has been converted to binary format.")
    print("New target distribution:")
    print(df['num'].value_counts())
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# ===================================
# STEP 2: HANDLE MISSING VALUES
# ===================================
print("\n🔧 STEP 2: HANDLING MISSING VALUES")
print("="*60)

# Check for missing values
missing_values = df.isnull().sum()
total_missing = missing_values.sum()
missing_percentage = (total_missing / (df.shape[0] * df.shape[1])) * 100

print(f"Total missing values: {total_missing}")
print(f"Missing percentage: {missing_percentage:.2f}%")

if total_missing > 0:
    print("\nMissing values per column:")
    print(missing_values[missing_values > 0])
    
    # Create missing values heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Missing Values Heatmap (Yellow = Missing)')
    plt.tight_layout()
    plt.show()
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Remove target from feature lists
    target_col = y.columns[0]
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    # Handle missing values
    if missing_percentage <= 1:
        # Row deletion for minimal missing data
        df_clean = df.dropna()
        print(f"✅ Used row deletion (minimal data loss: {missing_percentage:.2f}%)")
    else:
        df_clean = df.copy()
        # KNN imputation for numerical columns
        if len(numerical_cols) > 0:
            knn_imputer = KNNImputer(n_neighbors=5)
            df_clean[numerical_cols] = knn_imputer.fit_transform(df_clean[numerical_cols])
        
        # Mode imputation for categorical columns
        if len(categorical_cols) > 0:
            mode_imputer = SimpleImputer(strategy='most_frequent')
            df_clean[categorical_cols] = mode_imputer.fit_transform(df_clean[categorical_cols])
        
        print("✅ Used KNN imputation for numerical and mode for categorical")
    
    print(f"📊 Shape after handling missing values: {df_clean.shape}")
else:
    df_clean = df.copy()
    print("✅ No missing values found!")

# ===================================
# STEP 3: DATA ENCODING
# ===================================
print("\n🔤 STEP 3: DATA ENCODING")
print("="*60)

# Identify feature types
numerical_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()

# Remove target from features
target_col = y.columns[0]
if target_col in numerical_features:
    numerical_features.remove(target_col)
if target_col in categorical_features:
    categorical_features.remove(target_col)

print(f"📈 Numerical features ({len(numerical_features)}): {numerical_features}")
print(f"📊 Categorical features ({len(categorical_features)}): {categorical_features}")

# Start with clean dataframe
df_encoded = df_clean.copy()

# Handle categorical features encoding
if len(categorical_features) > 0:
    print(f"\n🔤 Encoding categorical features...")
    
    for col in categorical_features:
        unique_vals = df_encoded[col].nunique()
        print(f"  {col}: {unique_vals} unique values")
        
        if unique_vals == 2:
            # Binary encoding for binary categories
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            print(f"    ✅ Applied Label Encoding")
        else:
            # One-hot encoding for multi-class categories
            encoded_cols = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, encoded_cols], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)
            print(f"    ✅ Applied One-Hot Encoding ({len(encoded_cols.columns)} new columns)")
else:
    print("✅ No categorical features to encode!")

# Update feature lists after encoding
encoded_features = [col for col in df_encoded.columns if col != target_col]
print(f"\n📋 Total features after encoding: {len(encoded_features)}")

# ===================================
# STEP 4: FEATURE STANDARDIZATION
# ===================================
print("\n📏 STEP 4: FEATURE STANDARDIZATION")
print("="*60)

# Separate features and target
X_encoded = df_encoded.drop(target_col, axis=1)
y_encoded = df_encoded[target_col]

print(f"Features for scaling: {X_encoded.shape[1]}")
print(f"Target distribution:\n{y_encoded.value_counts().sort_index()}")

# Create both StandardScaler and MinMaxScaler versions
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

X_standard = scaler_standard.fit_transform(X_encoded)
X_minmax = scaler_minmax.fit_transform(X_encoded)

# Convert back to DataFrame
X_standard_df = pd.DataFrame(X_standard, columns=X_encoded.columns)
X_minmax_df = pd.DataFrame(X_minmax, columns=X_encoded.columns)

print("✅ Applied StandardScaler (mean=0, std=1)")
print("✅ Applied MinMaxScaler (range=0-1)")

# Compare scaling methods
print(f"\nStandardScaler stats:")
print(f"  Mean: {X_standard_df.mean().mean():.6f}")
print(f"  Std: {X_standard_df.std().mean():.6f}")
print(f"\nMinMaxScaler stats:")
print(f"  Min: {X_minmax_df.min().min():.6f}")
print(f"  Max: {X_minmax_df.max().max():.6f}")

# Use StandardScaler as default (better for most ML algorithms)
X_final = X_standard_df
scaler_used = "StandardScaler"
print(f"\n🎯 Using {scaler_used} for final dataset")
import pickle
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler_standard, f)
print("✅ StandardScaler object saved to 'models/scaler.pkl'")

# ===================================
# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
# ===================================
print("\n📊 STEP 5: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# 5.1: Dataset Overview
print("📋 DATASET OVERVIEW")
print("-" * 40)
print(f"Final dataset shape: {X_final.shape[0]} rows × {X_final.shape[1]} features")
print(f"Target variable: {target_col}")
print(f"Target classes: {sorted(y_encoded.unique())}")

# 5.2: Target Distribution Analysis
print(f"\n🎯 TARGET DISTRIBUTION")
print("-" * 40)
target_counts = y_encoded.value_counts().sort_index()
target_percentages = y_encoded.value_counts(normalize=True).sort_index() * 100

print("Class distribution:")
for class_val in sorted(y_encoded.unique()):
    count = target_counts[class_val]
    percent = target_percentages[class_val]
    print(f"  Class {class_val}: {count} samples ({percent:.1f}%)")

# Plot target distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Bar plot
target_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
ax1.set_title('Target Variable Distribution')
ax1.set_xlabel('Heart Disease')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=0)

# Pie chart
ax2.pie(target_counts.values, labels=[f'Class {i}' for i in target_counts.index], 
        autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
ax2.set_title('Target Variable Percentage')

plt.tight_layout()
plt.show()

# 5.3: Feature Distribution Analysis (Top 12 original numerical features)
original_numerical = [col for col in numerical_features if col in X_final.columns][:12]

if len(original_numerical) > 0:
    print(f"\n📈 FEATURE DISTRIBUTIONS (Top {len(original_numerical)} features)")
    print("-" * 40)
    
    # Histograms
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Feature Distributions (After Scaling)', fontsize=16)
    
    for i, col in enumerate(original_numerical):
        row = i // 4
        col_idx = i % 4
        
        if col in X_final.columns:
            axes[row, col_idx].hist(X_final[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[row, col_idx].set_title(f'{col}')
            axes[row, col_idx].set_xlabel('Value')
            axes[row, col_idx].set_ylabel('Frequency')
    
    # Hide empty subplots
    for i in range(len(original_numerical), 12):
        row = i // 4
        col_idx = i % 4
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# 5.4: Correlation Analysis
print(f"\n🔗 CORRELATION ANALYSIS")
print("-" * 40)

# Combine features and target for correlation
df_for_corr = X_final.copy()
df_for_corr[target_col] = y_encoded.values

# Calculate correlation matrix
correlation_matrix = df_for_corr.corr()

# Plot correlation heatmap
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Top correlations with target
target_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)
print(f"\nTop 10 features correlated with {target_col}:")
print(target_corr.head(11)[1:])  # Exclude self-correlation

# 5.5: Boxplots for Outlier Detection
print(f"\n📦 OUTLIER ANALYSIS (Boxplots)")
print("-" * 40)

# Select top correlated features for boxplot analysis
top_features = target_corr.head(9)[1:].index.tolist()  # Top 8 features

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Boxplots - Outlier Detection (Top Correlated Features)', fontsize=16)

for i, feature in enumerate(top_features):
    row = i // 4
    col = i % 4
    
    # Create boxplot grouped by target
    df_temp = pd.DataFrame({
        'feature': X_final[feature],
        'target': y_encoded.values
    })
    
    sns.boxplot(data=df_temp, x='target', y='feature', ax=axes[row, col])
    axes[row, col].set_title(f'{feature}')
    axes[row, col].set_xlabel('Heart Disease')

plt.tight_layout()
plt.show()

# 5.6: Statistical Summary
print(f"\n📊 STATISTICAL SUMMARY")
print("-" * 40)
print("Scaled features statistics:")
print(X_final.describe())

# ===================================
# SAVE PROCESSED DATA
# ===================================
print(f"\n💾 SAVING PROCESSED DATA")
print("="*60)

try:
    # Save final processed dataset
    final_dataset = X_final.copy()
    final_dataset[target_col] = y_encoded.values
    
    # Save files
    final_dataset.to_csv('data/heart_disease_final.csv', index=False)
    X_final.to_csv('data/X_processed.csv', index=False)
    y_encoded.to_csv('data/y_processed.csv', index=False)
    
    # Save preprocessing info
    preprocessing_info = {
        'original_shape': df.shape,
        'final_shape': final_dataset.shape,
        'missing_values_handled': total_missing,
        'encoding_applied': len(categorical_features) > 0,
        'scaling_method': scaler_used,
        'features_count': len(X_final.columns),
        'target_distribution': target_counts.to_dict()
    }
    
    import json
    with open('data/preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    
    print("✅ Files saved:")
    print("  📁 data/heart_disease_final.csv (Complete dataset)")
    print("  📁 data/X_processed.csv (Features)")
    print("  📁 data/y_processed.csv (Target)")
    print("  📁 data/preprocessing_info.json (Processing details)")
    
except Exception as e:
    print(f"⚠️ Error saving files: {e}")

# ===================================
# GENERATE COMPREHENSIVE REPORT
# ===================================
print(f"\n📝 GENERATING COMPREHENSIVE REPORT")
print("="*60)

report = f"""
HEART DISEASE DATA PREPROCESSING REPORT
=======================================

STEP 1 - DATA LOADING:
✅ Dataset: Heart Disease UCI (ID=45)
✅ Original shape: {df.shape}
✅ Target variable: {target_col}

STEP 2 - MISSING VALUES:
✅ Total missing values: {total_missing}
✅ Missing percentage: {missing_percentage:.2f}%
✅ Handling method: {"Row deletion" if missing_percentage <= 1 else "KNN + Mode imputation"}
✅ Shape after cleaning: {df_clean.shape}

STEP 3 - DATA ENCODING:
✅ Original categorical features: {len(categorical_features)}
✅ Encoding methods: Label Encoding (binary) + One-Hot Encoding (multi-class)
✅ Final feature count: {len(X_final.columns)}

STEP 4 - FEATURE SCALING:
✅ Scaling method: {scaler_used}
✅ Features scaled: {X_final.shape[1]}
✅ Mean after scaling: {X_final.mean().mean():.6f}
✅ Std after scaling: {X_final.std().mean():.6f}

STEP 5 - EXPLORATORY DATA ANALYSIS:
✅ Target distribution: {dict(target_counts)}
✅ Target percentages: {dict(target_percentages.round(1))}
✅ Correlation analysis completed
✅ Outlier detection completed
✅ Feature distributions analyzed

FINAL DATASET:
✅ Shape: {final_dataset.shape}
✅ Features: {X_final.shape[1]}
✅ Samples: {X_final.shape[0]}
✅ Ready for modeling: YES

FILES GENERATED:
✅ data/heart_disease_final.csv
✅ data/X_processed.csv
✅ data/y_processed.csv
✅ data/preprocessing_info.json

DELIVERABLE STATUS: ✅ COMPLETED
Dataset is cleaned and ready for modeling!
"""

print(report)

# Save report with proper handling
try:
    import os
    os.makedirs('results', exist_ok=True)
    
    # Clean the report content
    clean_report = report.replace('✅', '[OK]').replace('🚀', '[NEXT]').replace('⚠️', '[WARNING]')
    
    # Try multiple encoding methods
    report_saved = False
    
    # Method 1: UTF-8
    try:
        with open('results/preprocessing_complete_report.txt', 'w', encoding='utf-8') as f:
            f.write(clean_report)
            f.flush()
        
        # Verify file was written
        with open('results/preprocessing_complete_report.txt', 'r', encoding='utf-8') as f:
            check = f.read()
        
        if len(check) > 100:
            print("Complete report saved to: results/preprocessing_complete_report.txt")
            print(f"File size: {len(check)} characters")
            report_saved = True
    except:
        pass
    
    # Method 2: ASCII as fallback
    if not report_saved:
        try:
            with open('results/preprocessing_complete_report.txt', 'w', encoding='ascii', errors='ignore') as f:
                f.write(clean_report)
                f.flush()
            print("Report saved with ASCII encoding")
            report_saved = True
        except:
            pass
    
    # Method 3: Simple write
    if not report_saved:
        with open('results/preprocessing_complete_report.txt', 'w') as f:
            f.write(clean_report)
        print("Report saved with default encoding")
        
except Exception as e:
    print(f"Error saving report: {e}")
    # Print report to console as backup
    print("\n" + "="*60)
    print("REPORT CONTENT (since file save failed):")
    print("="*60)
    print(report)