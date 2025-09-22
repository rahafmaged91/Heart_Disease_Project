# ===================================
# FEATURE SELECTION - Complete Analysis
# Methods: Random Forest, XGBoost, RFE, Chi-Square
# ===================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("FEATURE SELECTION - COMPREHENSIVE ANALYSIS")
print("="*80)

# ===================================
# STEP 1: LOAD DATA
# ===================================
print("\n STEP 1: LOADING PROCESSED DATA")
print("="*60)

try:
    # Load both original processed data and PCA data
    X_processed = pd.read_csv('data/X_processed.csv')
    y_processed = pd.read_csv('data/y_processed.csv')
    X_pca = pd.read_csv('data/X_pca_transformed.csv')
    
    # Convert target to 1D array if needed
    if y_processed.shape[1] == 1:
        y_processed = y_processed.iloc[:, 0].values
    else:
        y_processed = y_processed.values
    
    print("[OK] Data loaded successfully!")
    print(f" Original features: {X_processed.shape}")
    print(f" PCA features: {X_pca.shape}")
    print(f" Target shape: {y_processed.shape}")
    print(f" Target classes: {np.unique(y_processed)}")
    
    # Display feature names
    print(f"\n Original features (first 10): {list(X_processed.columns[:10])}")
    print(f" PCA features: {list(X_pca.columns)}")
    
except Exception as e:
    print(f"[ERROR] Error loading data: {e}")
    print("[INFO] Make sure you've run preprocessing and PCA steps first!")
    exit()

# ===================================
# STEP 2: RANDOM FOREST FEATURE IMPORTANCE
# ===================================
print("\n STEP 2: RANDOM FOREST FEATURE IMPORTANCE")
print("="*60)

# Train Random Forest for feature importance
print("[PROCESS] Training Random Forest for feature importance...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_processed, y_processed)

# Get feature importance scores
rf_importance = rf_model.feature_importances_
rf_importance_df = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': rf_importance
}).sort_values('importance', ascending=False)

print("[OK] Random Forest feature importance calculated!")
print(f"\n[TOP 10] Top 10 features by Random Forest:")
print(rf_importance_df.head(10).to_string(index=False))

# Cross-validation score
rf_cv_score = cross_val_score(rf_model, X_processed, y_processed, cv=5, scoring='accuracy')
print(f"\n Random Forest CV Accuracy: {rf_cv_score.mean():.3f} ± {rf_cv_score.std():.3f}")

# ===================================
# STEP 3: XGBOOST FEATURE IMPORTANCE
# ===================================
print("\n STEP 3: XGBOOST FEATURE IMPORTANCE")
print("="*60)

# Train XGBoost for feature importance
print("[PROCESS] Training XGBoost for feature importance...")
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb_model.fit(X_processed, y_processed)

# Get feature importance scores
xgb_importance = xgb_model.feature_importances_
xgb_importance_df = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': xgb_importance
}).sort_values('importance', ascending=False)

print("[OK] XGBoost feature importance calculated!")
print(f"\n[TOP 10] Top 10 features by XGBoost:")
print(xgb_importance_df.head(10).to_string(index=False))

# Cross-validation score
xgb_cv_score = cross_val_score(xgb_model, X_processed, y_processed, cv=5, scoring='accuracy')
print(f"\n XGBoost CV Accuracy: {xgb_cv_score.mean():.3f} ± {xgb_cv_score.std():.3f}")

# ===================================
# STEP 4: RECURSIVE FEATURE ELIMINATION (RFE)
# ===================================
print("\n STEP 4: RECURSIVE FEATURE ELIMINATION (RFE)")
print("="*60)

# Apply RFE with Random Forest as estimator
n_features_to_select = min(15, X_processed.shape[1])  # Select top 15 features or all if less
print(f"[PROCESS] Applying RFE to select top {n_features_to_select} features...")

rfe_rf = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), 
             n_features_to_select=n_features_to_select, step=1)
rfe_rf.fit(X_processed, y_processed)

# Get RFE selected features
rfe_selected_features = X_processed.columns[rfe_rf.support_].tolist()
rfe_feature_ranking = rfe_rf.ranking_

rfe_results_df = pd.DataFrame({
    'feature': X_processed.columns,
    'selected': rfe_rf.support_,
    'ranking': rfe_feature_ranking
}).sort_values('ranking')

print("[OK] RFE completed!")
print(f"\n[SELECTED] RFE Selected Features ({len(rfe_selected_features)}):")
selected_df = rfe_results_df[rfe_results_df['selected'] == True]
print(selected_df[['feature', 'ranking']].to_string(index=False))

# Evaluate RFE selected features
X_rfe_selected = X_processed[rfe_selected_features]
rfe_cv_score = cross_val_score(RandomForestClassifier(random_state=42), 
                               X_rfe_selected, y_processed, cv=5, scoring='accuracy')
print(f"\n RFE Selected Features CV Accuracy: {rfe_cv_score.mean():.3f} ± {rfe_cv_score.std():.3f}")

# ===================================
# STEP 5: CHI-SQUARE TEST
# ===================================
print("\n STEP 5: CHI-SQUARE STATISTICAL TEST")
print("="*60)

# For Chi-square test, we need non-negative values
print("[PROCESS] Preparing data for Chi-square test...")

# Scale data to positive range using MinMaxScaler
scaler_chi2 = MinMaxScaler()
X_positive = scaler_chi2.fit_transform(X_processed)
X_positive_df = pd.DataFrame(X_positive, columns=X_processed.columns)

# Apply Chi-square test
k_best = min(15, X_processed.shape[1])  # Select top features
chi2_selector = SelectKBest(score_func=chi2, k=k_best)
chi2_selector.fit(X_positive, y_processed)

# Get Chi-square scores and p-values
chi2_scores = chi2_selector.scores_
chi2_pvalues = chi2_selector.pvalues_

chi2_results_df = pd.DataFrame({
    'feature': X_processed.columns,
    'chi2_score': chi2_scores,
    'p_value': chi2_pvalues,
    'selected': chi2_selector.get_support()
}).sort_values('chi2_score', ascending=False)

print("[OK] Chi-square test completed!")
print(f"\n[TOP 10] Top 10 features by Chi-square score:")
print(chi2_results_df.head(10)[['feature', 'chi2_score', 'p_value']].to_string(index=False))

# Get selected features
chi2_selected_features = chi2_results_df[chi2_results_df['selected']]['feature'].tolist()
print(f"\n[SELECTED] Chi-square selected features ({len(chi2_selected_features)}):")
for feature in chi2_selected_features:
    score = chi2_results_df[chi2_results_df['feature'] == feature]['chi2_score'].iloc[0]
    pval = chi2_results_df[chi2_results_df['feature'] == feature]['p_value'].iloc[0]
    print(f"  - {feature}: score={score:.2f}, p-value={pval:.4f}")

# ===================================
# STEP 6: F-SCORE (ANOVA) TEST
# ===================================
print("\n STEP 6: F-SCORE (ANOVA) STATISTICAL TEST")
print("="*60)

# Apply F-score test (works with negative values)
f_selector = SelectKBest(score_func=f_classif, k=k_best)
f_selector.fit(X_processed, y_processed)

# Get F-scores and p-values
f_scores = f_selector.scores_
f_pvalues = f_selector.pvalues_

f_results_df = pd.DataFrame({
    'feature': X_processed.columns,
    'f_score': f_scores,
    'p_value': f_pvalues,
    'selected': f_selector.get_support()
}).sort_values('f_score', ascending=False)

print("[OK] F-score test completed!")
print(f"\n[TOP 10] Top 10 features by F-score:")
print(f_results_df.head(10)[['feature', 'f_score', 'p_value']].to_string(index=False))

f_selected_features = f_results_df[f_results_df['selected']]['feature'].tolist()

# ===================================
# STEP 7: COMBINE ALL METHODS
# ===================================
print("\n STEP 7: COMBINING ALL FEATURE SELECTION METHODS")
print("="*60)

# Create comprehensive feature ranking
all_features = X_processed.columns.tolist()
feature_scores = pd.DataFrame({'feature': all_features})

# Add rankings from each method
feature_scores['rf_importance'] = feature_scores['feature'].map(
    dict(zip(rf_importance_df['feature'], rf_importance_df['importance']))
)
feature_scores['xgb_importance'] = feature_scores['feature'].map(
    dict(zip(xgb_importance_df['feature'], xgb_importance_df['importance']))
)
feature_scores['rfe_ranking'] = feature_scores['feature'].map(
    dict(zip(rfe_results_df['feature'], rfe_results_df['ranking']))
)
feature_scores['chi2_score'] = feature_scores['feature'].map(
    dict(zip(chi2_results_df['feature'], chi2_results_df['chi2_score']))
)
feature_scores['f_score'] = feature_scores['feature'].map(
    dict(zip(f_results_df['feature'], f_results_df['f_score']))
)

# Normalize scores to 0-1 range for fair comparison
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
feature_scores['rf_importance_norm'] = scaler.fit_transform(feature_scores[['rf_importance']])
feature_scores['xgb_importance_norm'] = scaler.fit_transform(feature_scores[['xgb_importance']])
feature_scores['rfe_ranking_norm'] = 1 - scaler.fit_transform(feature_scores[['rfe_ranking']])  # Lower rank = better
feature_scores['chi2_score_norm'] = scaler.fit_transform(feature_scores[['chi2_score']])
feature_scores['f_score_norm'] = scaler.fit_transform(feature_scores[['f_score']])

# Calculate composite score
feature_scores['composite_score'] = (
    feature_scores['rf_importance_norm'] + 
    feature_scores['xgb_importance_norm'] + 
    feature_scores['rfe_ranking_norm'] + 
    feature_scores['chi2_score_norm'] + 
    feature_scores['f_score_norm']
) / 5

# Sort by composite score
feature_scores = feature_scores.sort_values('composite_score', ascending=False)

print("[OK] Composite feature ranking calculated!")
print(f"\n[TOP 15] TOP 15 FEATURES BY COMPOSITE SCORE:")
print(feature_scores.head(15)[['feature', 'composite_score']].to_string(index=False))

# Select final features based on composite score
n_final_features = min(12, len(all_features))  # Select top 12 features
final_selected_features = feature_scores.head(n_final_features)['feature'].tolist()

print(f"\n[FINAL] FINAL SELECTED FEATURES ({n_final_features}):")
for i, feature in enumerate(final_selected_features, 1):
    score = feature_scores[feature_scores['feature'] == feature]['composite_score'].iloc[0]
    print(f"  {i:2d}. {feature} (score: {score:.3f})")

# ===================================
# STEP 8: VISUALIZATIONS
# ===================================
print("\n STEP 8: FEATURE IMPORTANCE VISUALIZATIONS")
print("="*60)

# 8.1: Feature Importance Comparison (Top 15 features)
top_15_features = feature_scores.head(15)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Feature Selection Methods Comparison', fontsize=16, fontweight='bold')

# Random Forest Importance
rf_top_15 = rf_importance_df.head(15)
ax1.barh(range(len(rf_top_15)), rf_top_15['importance'], color='skyblue')
ax1.set_yticks(range(len(rf_top_15)))
ax1.set_yticklabels(rf_top_15['feature'])
ax1.set_xlabel('Importance Score')
ax1.set_title('Random Forest Feature Importance')
ax1.invert_yaxis()

# XGBoost Importance
xgb_top_15 = xgb_importance_df.head(15)
ax2.barh(range(len(xgb_top_15)), xgb_top_15['importance'], color='lightcoral')
ax2.set_yticks(range(len(xgb_top_15)))
ax2.set_yticklabels(xgb_top_15['feature'])
ax2.set_xlabel('Importance Score')
ax2.set_title('XGBoost Feature Importance')
ax2.invert_yaxis()

# Chi-square Scores
chi2_top_15 = chi2_results_df.head(15)
ax3.barh(range(len(chi2_top_15)), chi2_top_15['chi2_score'], color='lightgreen')
ax3.set_yticks(range(len(chi2_top_15)))
ax3.set_yticklabels(chi2_top_15['feature'])
ax3.set_xlabel('Chi-square Score')
ax3.set_title('Chi-square Test Scores')
ax3.invert_yaxis()

# Composite Scores
ax4.barh(range(len(top_15_features)), top_15_features['composite_score'], color='gold')
ax4.set_yticks(range(len(top_15_features)))
ax4.set_yticklabels(top_15_features['feature'])
ax4.set_xlabel('Composite Score')
ax4.set_title('Composite Feature Ranking')
ax4.invert_yaxis()

plt.tight_layout()
plt.show()

# 8.2: Feature Selection Methods Overlap
methods = ['Random Forest', 'XGBoost', 'RFE', 'Chi-square', 'F-score']
top_10_sets = [
    set(rf_importance_df.head(10)['feature']),
    set(xgb_importance_df.head(10)['feature']),
    set(rfe_selected_features[:10]),
    set(chi2_selected_features[:10]),
    set(f_selected_features[:10])
]

# Create overlap matrix
overlap_matrix = np.zeros((len(methods), len(methods)))
for i, set1 in enumerate(top_10_sets):
    for j, set2 in enumerate(top_10_sets):
        if i != j:
            overlap = len(set1.intersection(set2))
            overlap_matrix[i][j] = int(overlap)
        else:
            overlap_matrix[i][j] = len(set1)

plt.figure(figsize=(10, 8))
sns.heatmap(overlap_matrix.astype(int), annot=True, fmt='d', cmap='Blues',
            xticklabels=methods, yticklabels=methods)
plt.title('Feature Selection Methods Overlap\n(Number of Common Features in Top 10)')
plt.tight_layout()
plt.show()

# 8.3: Correlation between different importance scores
plt.figure(figsize=(15, 10))

# Handle any NaN values and ensure proper correlation calculation
correlation_data = feature_scores[['rf_importance', 'xgb_importance', 'chi2_score', 'f_score']].fillna(0)
correlation_matrix = correlation_data.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
plt.title('Correlation Between Different Feature Importance Methods')
plt.tight_layout()
plt.show()

# 8.4: Top features comparison across methods
plt.figure(figsize=(16, 10))

# Get top 8 features from each method
top_n = 8
methods_top_features = {
    'Random Forest': rf_importance_df.head(top_n)['feature'].tolist(),
    'XGBoost': xgb_importance_df.head(top_n)['feature'].tolist(),
    'RFE': rfe_selected_features[:top_n],
    'Chi-square': chi2_selected_features[:top_n],
    'F-score': f_selected_features[:top_n],
    'Composite': final_selected_features[:top_n]
}

# Create a presence matrix
all_top_features = set()
for features in methods_top_features.values():
    all_top_features.update(features)
all_top_features = sorted(list(all_top_features))

presence_matrix = np.zeros((len(all_top_features), len(methods_top_features)))
for j, (method, features) in enumerate(methods_top_features.items()):
    for i, feature in enumerate(all_top_features):
        if feature in features:
            presence_matrix[i, j] = 1

plt.imshow(presence_matrix, cmap='RdYlBu_r', aspect='auto')
plt.yticks(range(len(all_top_features)), all_top_features, rotation=0, fontsize=9)
plt.xticks(range(len(methods_top_features)), list(methods_top_features.keys()), rotation=45, ha='right')
plt.title(f'Top {top_n} Features Selection by Different Methods\n(Blue = Selected, Red = Not Selected)')
plt.colorbar(label='Selected (1) / Not Selected (0)')

# Add text annotations
for i in range(len(all_top_features)):
    for j in range(len(methods_top_features)):
        text = 'OK' if presence_matrix[i, j] == 1 else 'X'
        color = 'white' if presence_matrix[i, j] == 1 else 'black'
        plt.text(j, i, text, ha='center', va='center', color=color, fontweight='bold')

plt.tight_layout()
plt.show()

# ===================================
# STEP 9: EVALUATE FEATURE SUBSETS
# ===================================
print("\n STEP 9: EVALUATING DIFFERENT FEATURE SUBSETS")
print("="*60)

# Test different numbers of features
feature_counts = [5, 8, 10, 12, 15, min(20, len(all_features))]
evaluation_results = []

for n_features in feature_counts:
    if n_features <= len(all_features):
        # Select top N features
        selected_features = feature_scores.head(n_features)['feature'].tolist()
        X_selected = X_processed[selected_features]
        
        # Evaluate with Random Forest
        rf_eval = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_scores = cross_val_score(rf_eval, X_selected, y_processed, cv=5, scoring='accuracy')
        
        # Evaluate with XGBoost
        xgb_eval = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        xgb_scores = cross_val_score(xgb_eval, X_selected, y_processed, cv=5, scoring='accuracy')
        
        evaluation_results.append({
            'n_features': n_features,
            'rf_accuracy': rf_scores.mean(),
            'rf_std': rf_scores.std(),
            'xgb_accuracy': xgb_scores.mean(),
            'xgb_std': xgb_scores.std()
        })

evaluation_df = pd.DataFrame(evaluation_results)
print("[EVALUATION] Feature subset evaluation:")
print(evaluation_df.to_string(index=False))

# Plot evaluation results
plt.figure(figsize=(12, 6))
plt.errorbar(evaluation_df['n_features'], evaluation_df['rf_accuracy'], 
             yerr=evaluation_df['rf_std'], marker='o', label='Random Forest', linewidth=2)
plt.errorbar(evaluation_df['n_features'], evaluation_df['xgb_accuracy'], 
             yerr=evaluation_df['xgb_std'], marker='s', label='XGBoost', linewidth=2)
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Model Performance vs Number of Features')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Find optimal number of features
optimal_rf = evaluation_df.loc[evaluation_df['rf_accuracy'].idxmax()]
optimal_xgb = evaluation_df.loc[evaluation_df['xgb_accuracy'].idxmax()]

print(f"\n[OPTIMAL] OPTIMAL FEATURE COUNTS:")
print(f"  Random Forest: {optimal_rf['n_features']} features (accuracy: {optimal_rf['rf_accuracy']:.3f})")
print(f"  XGBoost: {optimal_xgb['n_features']} features (accuracy: {optimal_xgb['xgb_accuracy']:.3f})")

# ===================================
# STEP 10: SAVE RESULTS
# ===================================
print("\n STEP 10: SAVING FEATURE SELECTION RESULTS")
print("="*60)

try:
    # Save selected features dataset
    X_final_selected = X_processed[final_selected_features]
    X_final_selected.to_csv('data/X_selected_features.csv', index=False)
    
    # Save complete dataset with selected features and target
    final_dataset = X_final_selected.copy()
    final_dataset['target'] = y_processed
    final_dataset.to_csv('data/heart_disease_selected_features.csv', index=False)
    
    # Save feature importance rankings
    feature_scores.to_csv('results/feature_importance_rankings.csv', index=False)
    
    # Save evaluation results
    evaluation_df.to_csv('results/feature_evaluation_results.csv', index=False)
    
    # Save feature selection info with proper encoding
    selection_info = {
        'total_original_features': len(all_features),
        'final_selected_features': len(final_selected_features),
        'selected_feature_names': final_selected_features,
        'selection_methods_used': ['Random Forest', 'XGBoost', 'RFE', 'Chi-square', 'F-score'],
        'optimal_feature_counts': {
            'random_forest': int(optimal_rf['n_features']),
            'xgboost': int(optimal_xgb['n_features'])
        },
        'performance_with_selected_features': {
            'random_forest_accuracy': float(optimal_rf['rf_accuracy']),
            'xgboost_accuracy': float(optimal_xgb['xgb_accuracy'])
        }
    }
    
    import json
    with open('data/feature_selection_info.json', 'w', encoding='utf-8') as f:
        json.dump(selection_info, f, indent=2)
    
    print("[OK] Feature selection results saved:")
    print("  data/X_selected_features.csv (Selected features only)")
    print("  data/heart_disease_selected_features.csv (Complete dataset)")
    print("  results/feature_importance_rankings.csv (All rankings)")
    print("  results/feature_evaluation_results.csv (Evaluation results)")
    print("  data/feature_selection_info.json (Selection info)")
    
except Exception as e:
    print(f"[ERROR] Error saving results: {e}")

# ===================================
# STEP 11: GENERATE REPORT
# ===================================
print("\n STEP 11: GENERATING FEATURE SELECTION REPORT")
print("="*60)

feature_selection_report = f"""
FEATURE SELECTION COMPREHENSIVE REPORT
======================================

ORIGINAL DATA:
- Total features: {len(all_features)}
- Samples: {X_processed.shape[0]}
- Target classes: {list(np.unique(y_processed))}

METHODS APPLIED:
- Random Forest Importance
- XGBoost Importance  
- Recursive Feature Elimination (RFE)
- Chi-square Statistical Test
- F-score (ANOVA) Statistical Test

PERFORMANCE COMPARISON:
- Random Forest CV: {rf_cv_score.mean():.3f} ± {rf_cv_score.std():.3f}
- XGBoost CV: {xgb_cv_score.mean():.3f} ± {xgb_cv_score.std():.3f}
- RFE Selected CV: {rfe_cv_score.mean():.3f} ± {rfe_cv_score.std():.3f}

FINAL SELECTION:
- Features selected: {len(final_selected_features)}
- Selection method: Composite scoring (average of all methods)
- Reduction ratio: {(1 - len(final_selected_features)/len(all_features))*100:.1f}%

SELECTED FEATURES:
{chr(10).join([f"  {i:2d}. {feature}" for i, feature in enumerate(final_selected_features, 1)])}

OPTIMAL FEATURE COUNTS:
- Random Forest optimal: {optimal_rf['n_features']} features ({optimal_rf['rf_accuracy']:.3f} accuracy)
- XGBoost optimal: {optimal_xgb['n_features']} features ({optimal_xgb['xgb_accuracy']:.3f} accuracy)

FILES GENERATED:
- data/X_selected_features.csv (Selected features dataset)
- data/heart_disease_selected_features.csv (Complete selected dataset)
- results/feature_importance_rankings.csv (All method rankings)
- results/feature_evaluation_results.csv (Performance evaluation)
- data/feature_selection_info.json (Selection metadata)

DELIVERABLES STATUS:
- Reduced dataset with selected key features: COMPLETED
- Feature importance ranking visualization: COMPLETED
- Ready for next step: Model Training

NEXT STEPS:
- Supervised Learning with selected features
- Compare performance: All features vs Selected features vs PCA features
- Hyperparameter tuning on selected features
"""

print(feature_selection_report)

# Save report with proper encoding
try:
    with open('results/feature_selection_report.txt', 'w', encoding='utf-8') as f:
        f.write(feature_selection_report)
    print("[OK] Feature selection report saved to: results/feature_selection_report.txt")
except Exception as e:
    print(f"[ERROR] Error saving report: {e}")
    # Fallback: save without special characters
    try:
        clean_report = feature_selection_report.replace('✅', '[OK]').replace('🚀', '[NEXT]')
        with open('results/feature_selection_report.txt', 'w') as f:
            f.write(clean_report)
        print("[OK] Feature selection report saved with clean formatting")
    except:
        print("[ERROR] Could not save report")

print(f"\n[SUCCESS] FEATURE SELECTION COMPLETED SUCCESSFULLY!")
print("="*80)
print("[OK] Best features identified using multiple methods")
print("[OK] All deliverables generated")
print("[NEXT] Ready for next phase: Supervised Learning!")
print("="*80)