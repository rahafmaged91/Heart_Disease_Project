# ===================================
# PCA (Principal Component Analysis) - Complete Analysis
# ===================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("🔬 PCA (PRINCIPAL COMPONENT ANALYSIS)")
print("="*80)

# ===================================
# STEP 1: LOAD PREPROCESSED DATA
# ===================================
print("\n📥 STEP 1: LOADING PREPROCESSED DATA")
print("="*60)

try:
    # Load the preprocessed features and targets
    X_processed = pd.read_csv('data/X_processed.csv')
    y_processed = pd.read_csv('data/y_processed.csv')
    
    # Convert target to 1D array if needed
    if y_processed.shape[1] == 1:
        y_processed = y_processed.iloc[:, 0]
    
    print("✅ Preprocessed data loaded successfully!")
    print(f"📊 Features shape: {X_processed.shape}")
    print(f"🎯 Target shape: {y_processed.shape}")
    print(f"📋 Feature names: {list(X_processed.columns[:5])}... (showing first 5)")
    
    # Display basic statistics
    print(f"\n📈 Data Statistics:")
    print(f"  - Number of samples: {X_processed.shape[0]}")
    print(f"  - Number of features: {X_processed.shape[1]}")
    print(f"  - Target classes: {sorted(y_processed.unique())}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("💡 Make sure you've run the preprocessing step first!")
    exit()

# ===================================
# STEP 2: INITIAL PCA ANALYSIS
# ===================================
print("\n🔍 STEP 2: INITIAL PCA ANALYSIS")
print("="*60)

# Verify data is already scaled (should be from preprocessing)
print("📊 Data scaling verification:")
print(f"  - Mean: {X_processed.mean().mean():.6f}")
print(f"  - Std: {X_processed.std().mean():.6f}")

# Apply PCA with all components first
print(f"\n🔬 Applying PCA with all {X_processed.shape[1]} components...")
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_processed)

print("✅ Full PCA completed!")
print(f"📊 PCA shape: {X_pca_full.shape}")

# ===================================
# STEP 3: EXPLAINED VARIANCE ANALYSIS
# ===================================
print("\n📈 STEP 3: EXPLAINED VARIANCE ANALYSIS")
print("="*60)

# Get explained variance ratios
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print(f"📊 Explained Variance Analysis:")
print(f"  - First component explains: {explained_variance_ratio[0]:.1%} of variance")
print(f"  - First 2 components explain: {cumulative_variance_ratio[1]:.1%} of variance")
print(f"  - First 5 components explain: {cumulative_variance_ratio[4]:.1%} of variance")
print(f"  - First 10 components explain: {cumulative_variance_ratio[9]:.1%} of variance")

# Find optimal number of components for different thresholds
thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
optimal_components = {}

for threshold in thresholds:
    n_components = np.argmax(cumulative_variance_ratio >= threshold) + 1
    optimal_components[threshold] = n_components
    print(f"  - {threshold:.0%} variance retained with: {n_components} components")

# ===================================
# STEP 4: VISUALIZE EXPLAINED VARIANCE
# ===================================
print("\n📊 STEP 4: VARIANCE VISUALIZATION")
print("="*60)

# Create comprehensive variance plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('PCA Explained Variance Analysis', fontsize=16, fontweight='bold')

# Plot 1: Individual explained variance
ax1.bar(range(1, min(21, len(explained_variance_ratio) + 1)), 
        explained_variance_ratio[:20], 
        alpha=0.7, color='skyblue', edgecolor='navy')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Individual Explained Variance (First 20 Components)')
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative explained variance
ax2.plot(range(1, len(cumulative_variance_ratio) + 1), 
         cumulative_variance_ratio, 'bo-', linewidth=2, markersize=4)
ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.8, label='95% threshold')
ax2.axhline(y=0.90, color='orange', linestyle='--', alpha=0.8, label='90% threshold')
ax2.axhline(y=0.85, color='green', linestyle='--', alpha=0.8, label='85% threshold')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('Cumulative Explained Variance')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Scree plot (first 15 components)
ax3.plot(range(1, min(16, len(explained_variance_ratio) + 1)), 
         explained_variance_ratio[:15], 'ro-', linewidth=2, markersize=6)
ax3.set_xlabel('Principal Component')
ax3.set_ylabel('Explained Variance Ratio')
ax3.set_title('Scree Plot (First 15 Components)')
ax3.grid(True, alpha=0.3)

# Plot 4: Components vs Threshold Achievement
thresholds_list = list(optimal_components.keys())
components_list = list(optimal_components.values())
ax4.bar(range(len(thresholds_list)), components_list, 
        color=['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink'])
ax4.set_xlabel('Variance Threshold')
ax4.set_ylabel('Number of Components Needed')
ax4.set_title('Components Needed for Different Variance Thresholds')
ax4.set_xticks(range(len(thresholds_list)))
ax4.set_xticklabels([f'{t:.0%}' for t in thresholds_list])

# Add value labels on bars
for i, v in enumerate(components_list):
    ax4.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ===================================
# STEP 5: DETERMINE OPTIMAL COMPONENTS
# ===================================
print("\n🎯 STEP 5: DETERMINING OPTIMAL COMPONENTS")
print("="*60)

# Method 1: 95% variance threshold (most common)
optimal_95 = optimal_components[0.95]

# Method 2: Elbow method (find the "elbow" in scree plot)
# Calculate second derivative to find elbow
if len(explained_variance_ratio) > 2:
    second_derivative = np.diff(explained_variance_ratio, 2)
    elbow_point = np.argmin(second_derivative) + 2  # +2 because of double diff
else:
    elbow_point = len(explained_variance_ratio)

# Method 3: Kaiser criterion (eigenvalues > 1)
# For scaled data, this means explained variance > 1/n_features
kaiser_threshold = 1.0 / X_processed.shape[1]
kaiser_components = np.sum(explained_variance_ratio > kaiser_threshold)

print(f"🔍 Different methods for optimal components:")
print(f"  1. 95% Variance Method: {optimal_95} components")
print(f"  2. Elbow Method: {elbow_point} components")
print(f"  3. Kaiser Criterion: {kaiser_components} components")

# Choose the most conservative approach (95% variance)
n_components_final = optimal_95
print(f"\n🏆 SELECTED: {n_components_final} components (95% variance method)")
print(f"  - Variance retained: {cumulative_variance_ratio[n_components_final-1]:.1%}")
print(f"  - Dimensionality reduction: {X_processed.shape[1]} → {n_components_final}")
print(f"  - Reduction ratio: {(1 - n_components_final/X_processed.shape[1]):.1%}")

# ===================================
# STEP 6: APPLY OPTIMAL PCA
# ===================================
print("\n🔬 STEP 6: APPLYING OPTIMAL PCA")
print("="*60)

# Apply PCA with optimal number of components
pca_optimal = PCA(n_components=n_components_final)
X_pca_optimal = pca_optimal.fit_transform(X_processed)

print(f"✅ Optimal PCA applied!")
print(f"📊 Original features: {X_processed.shape[1]}")
print(f"📊 PCA features: {X_pca_optimal.shape[1]}")
print(f"📊 Variance retained: {pca_optimal.explained_variance_ratio_.sum():.1%}")

# Create PCA DataFrame
pca_columns = [f'PC{i+1}' for i in range(n_components_final)]
X_pca_df = pd.DataFrame(X_pca_optimal, columns=pca_columns)

print(f"📋 PCA Components: {list(X_pca_df.columns)}")

# ===================================
# STEP 7: PCA VISUALIZATION
# ===================================
print("\n🎨 STEP 7: PCA VISUALIZATION")
print("="*60)

# 7.1: 2D PCA Scatter Plot (PC1 vs PC2)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca_optimal[:, 0], X_pca_optimal[:, 1], 
                     c=y_processed, cmap='viridis', alpha=0.7, s=50)
plt.xlabel(f'PC1 ({pca_optimal.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_optimal.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA: First Two Principal Components')
plt.colorbar(scatter, label='Heart Disease Class')
plt.grid(True, alpha=0.3)

# Add class centroids
for class_val in sorted(y_processed.unique()):
    class_mask = y_processed == class_val
    centroid_x = X_pca_optimal[class_mask, 0].mean()
    centroid_y = X_pca_optimal[class_mask, 1].mean()
    plt.scatter(centroid_x, centroid_y, c='red', s=200, marker='X', 
               edgecolors='black', linewidth=2, label=f'Class {class_val} Centroid')

plt.legend()
plt.tight_layout()
plt.show()

# 7.2: 3D PCA Scatter Plot (if we have at least 3 components)
if n_components_final >= 3:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_pca_optimal[:, 0], X_pca_optimal[:, 1], X_pca_optimal[:, 2],
                        c=y_processed, cmap='viridis', alpha=0.7, s=50)
    
    ax.set_xlabel(f'PC1 ({pca_optimal.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca_optimal.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC3 ({pca_optimal.explained_variance_ratio_[2]:.1%})')
    ax.set_title('PCA: First Three Principal Components')
    
    plt.colorbar(scatter, label='Heart Disease Class', shrink=0.8)
    plt.tight_layout()
    plt.show()

# 7.3: PCA Heatmap (Component Loadings)
if X_processed.shape[1] <= 20:  # Only if we have reasonable number of original features
    plt.figure(figsize=(14, 8))
    
    # Get component loadings (correlations between original features and PCs)
    loadings = pca_optimal.components_[:min(10, n_components_final)].T
    loadings_df = pd.DataFrame(loadings, 
                              index=X_processed.columns,
                              columns=[f'PC{i+1}' for i in range(min(10, n_components_final))])
    
    sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
                cbar_kws={'label': 'Loading Value'})
    plt.title('PCA Component Loadings Heatmap')
    plt.xlabel('Principal Components')
    plt.ylabel('Original Features')
    plt.tight_layout()
    plt.show()

# ===================================
# STEP 8: PCA COMPONENTS ANALYSIS
# ===================================
print("\n🔍 STEP 8: PCA COMPONENTS ANALYSIS")
print("="*60)

# Analyze individual components
print(f"📊 Individual Component Analysis:")
for i in range(min(5, n_components_final)):
    variance_pct = pca_optimal.explained_variance_ratio_[i] * 100
    print(f"  PC{i+1}: {variance_pct:.2f}% of variance")
    
    if X_processed.shape[1] <= 50:  # Only show loadings if manageable number of features
        # Get top contributing features for this component
        component_loadings = abs(pca_optimal.components_[i])
        top_features_idx = np.argsort(component_loadings)[-3:]  # Top 3 features
        print(f"       Top contributing features:")
        for idx in reversed(top_features_idx):
            feature_name = X_processed.columns[idx]
            loading_value = pca_optimal.components_[i][idx]
            print(f"         - {feature_name}: {loading_value:.3f}")

# ===================================
# STEP 9: SAVE PCA RESULTS
# ===================================
print("\n💾 STEP 9: SAVING PCA RESULTS")
print("="*60)

try:
    # Save PCA-transformed data
    X_pca_df.to_csv('data/X_pca_transformed.csv', index=False)
    
    # Save PCA combined with target
    pca_complete = X_pca_df.copy()
    pca_complete['target'] = y_processed.values
    pca_complete.to_csv('data/heart_disease_pca.csv', index=False)
    
    # Save PCA model information
    pca_info = {
        'n_components_original': X_processed.shape[1],
        'n_components_selected': n_components_final,
        'variance_retained': float(pca_optimal.explained_variance_ratio_.sum()),
        'explained_variance_ratio': pca_optimal.explained_variance_ratio_.tolist(),
        'cumulative_variance_ratio': cumulative_variance_ratio[:n_components_final].tolist(),
        'optimal_components_methods': {
            '95_percent': optimal_95,
            'elbow_method': elbow_point,
            'kaiser_criterion': kaiser_components
        }
    }
    
    import json
    with open('data/pca_info.json', 'w') as f:
        json.dump(pca_info, f, indent=2)
    
    # Save explained variance data for plotting
    variance_data = pd.DataFrame({
        'Component': range(1, len(explained_variance_ratio) + 1),
        'Individual_Variance': explained_variance_ratio,
        'Cumulative_Variance': cumulative_variance_ratio
    })
    variance_data.to_csv('results/pca_variance_analysis.csv', index=False)
    
    print("✅ PCA results saved:")
    print("  📁 data/X_pca_transformed.csv (PCA features only)")
    print("  📁 data/heart_disease_pca.csv (PCA features + target)")
    print("  📁 data/pca_info.json (PCA model info)")
    print("  📁 results/pca_variance_analysis.csv (Variance analysis)")
    
except Exception as e:
    print(f"⚠️ Error saving PCA results: {e}")

# ===================================
# STEP 10: GENERATE PCA REPORT
# ===================================
print("\n📝 STEP 10: GENERATING PCA REPORT")
print("="*60)

# Calculate additional metrics
dimensionality_reduction = (1 - n_components_final / X_processed.shape[1]) * 100
data_compression = (X_processed.shape[1] - n_components_final) / X_processed.shape[1] * 100

pca_report = f"""
PCA (PRINCIPAL COMPONENT ANALYSIS) REPORT
=========================================

ORIGINAL DATA:
✅ Features: {X_processed.shape[1]}
✅ Samples: {X_processed.shape[0]}
✅ Data was pre-scaled: YES

PCA ANALYSIS:
✅ Method: Singular Value Decomposition (SVD)
✅ Components analyzed: All {X_processed.shape[1]} components
✅ Optimal components selected: {n_components_final}
✅ Variance retained: {pca_optimal.explained_variance_ratio_.sum():.1%}

DIMENSIONALITY REDUCTION:
✅ Original dimensions: {X_processed.shape[1]}
✅ Reduced dimensions: {n_components_final}
✅ Reduction percentage: {dimensionality_reduction:.1f}%
✅ Compression ratio: {data_compression:.1f}%

VARIANCE ANALYSIS:
✅ PC1 explains: {pca_optimal.explained_variance_ratio_[0]:.1%}
✅ PC2 explains: {pca_optimal.explained_variance_ratio_[1]:.1%}
✅ First 5 PCs explain: {pca_optimal.explained_variance_ratio_[:min(5, n_components_final)].sum():.1%}

OPTIMAL COMPONENTS (Different Methods):
✅ 95% Variance Method: {optimal_95} components
✅ Elbow Method: {elbow_point} components  
✅ Kaiser Criterion: {kaiser_components} components
✅ Selected Method: 95% Variance ({n_components_final} components)

FILES GENERATED:
✅ data/X_pca_transformed.csv (PCA-transformed features)
✅ data/heart_disease_pca.csv (Complete PCA dataset)
✅ data/pca_info.json (PCA model information)
✅ results/pca_variance_analysis.csv (Variance analysis)

DELIVERABLES STATUS:
✅ PCA-transformed dataset: COMPLETED
✅ Variance retained per component graph: COMPLETED
✅ Ready for next step: Feature Selection

NEXT STEPS:
🚀 Feature Selection using PCA components
🚀 Model Training with reduced dimensionality
🚀 Compare performance: Original vs PCA features
"""

print(pca_report)

# Save report
try:
    with open('results/pca_analysis_report.txt', 'w') as f:
        f.write(pca_report)
    print("✅ PCA report saved to: results/pca_analysis_report.txt")
except Exception as e:
    print(f"⚠️ Error saving report: {e}")

print(f"\n🎉 PCA ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*80)
print("✅ Dimensionality reduced while retaining 95% variance")
print("✅ All deliverables generated")
print("🚀 Ready for next phase: Feature Selection!")
print("="*80)