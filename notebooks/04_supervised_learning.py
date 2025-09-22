# ===================================
# SUPERVISED LEARNING - CLASSIFICATION MODELS
# Models: Logistic Regression, Decision Tree, Random Forest, SVM
# ===================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix, roc_curve, auc,
                           roc_auc_score, precision_recall_curve)
from sklearn.preprocessing import label_binarize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

print("SUPERVISED LEARNING - CLASSIFICATION MODELS")
print("="*80)

# ===================================
# STEP 1: LOAD DATA
# ===================================
print("\n STEP 1: LOADING PROCESSED DATA")
print("="*60)

try:
    # Load the selected features from feature selection step
    X_selected = pd.read_csv('data/X_selected_features.csv')
    y_processed = pd.read_csv('data/y_processed.csv')
    
    # Also load original processed data and PCA data for comparison
    X_processed = pd.read_csv('data/X_processed.csv')
    X_pca = pd.read_csv('data/X_pca_transformed.csv')
    
    # Convert target to 1D array if needed
    if y_processed.shape[1] == 1:
        y = y_processed.iloc[:, 0].values
    else:
        y = y_processed.values.ravel()
    
    print("[OK] Data loaded successfully!")
    print(f" Selected features: {X_selected.shape}")
    print(f" Original features: {X_processed.shape}")
    print(f" PCA features: {X_pca.shape}")
    print(f" Target shape: {y.shape}")
    print(f" Target classes: {np.unique(y)}")
    
    # Display feature names
    print(f"\n Selected features ({len(X_selected.columns)}):")
    print(list(X_selected.columns))
    
except Exception as e:
    print(f"[ERROR] Error loading data: {e}")
    print("[INFO] Make sure you've run preprocessing and feature selection steps first!")
    exit()

# ===================================
# STEP 2: DATA SPLITTING
# ===================================
print("\n STEP 2: SPLITTING DATA INTO TRAIN/TEST SETS")
print("="*60)

# Split selected features dataset (primary)
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.20, random_state=42, stratify=y
)

print("[OK] Data split completed!")
print(f" Training set: {X_train.shape}")
print(f" Testing set: {X_test.shape}")
print(f" Training target: {y_train.shape}")
print(f" Testing target: {y_test.shape}")

# Check class distribution
print(f"\n Class distribution in training set:")
train_counts = pd.Series(y_train).value_counts().sort_index()
print(train_counts)
print(f"Training class percentages: {train_counts / len(y_train) * 100}")

print(f"\n Class distribution in testing set:")
test_counts = pd.Series(y_test).value_counts().sort_index()
print(test_counts)
print(f"Testing class percentages: {test_counts / len(y_test) * 100}")

# Also split other datasets for comparison later
X_train_orig, X_test_orig, _, _ = train_test_split(
    X_processed, y, test_size=0.20, random_state=42, stratify=y
)

X_train_pca, X_test_pca, _, _ = train_test_split(
    X_pca, y, test_size=0.20, random_state=42, stratify=y
)

# ===================================
# STEP 3: INITIALIZE MODELS
# ===================================
print("\n STEP 3: INITIALIZING CLASSIFICATION MODELS")
print("="*60)

# Define models with their parameters
models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, 
        max_iter=1000,
        solver='liblinear'  # Good for small datasets
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=42,
        max_depth=10,  # Prevent overfitting
        min_samples_split=20,
        min_samples_leaf=10
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=5,
        n_jobs=-1
    ),
    'Support Vector Machine': SVC(
        kernel='rbf',
        random_state=42,
        probability=True,  # Enable probability estimates for ROC
        C=1.0,
        gamma='scale'
    )
}

print("[OK] Models initialized:")
for name, model in models.items():
    print(f"  - {name}: {type(model).__name__}")

# ===================================
# STEP 4: TRAIN MODELS AND EVALUATE
# ===================================
print("\n STEP 4: TRAINING MODELS AND EVALUATION")
print("="*60)

# Storage for results
results = {}
trained_models = {}
training_times = {}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n[PROCESS] Training {name}...")
    
    # Track training time
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    training_times[name] = training_time
    
    # Store trained model
    trained_models[name] = model
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Get prediction probabilities for ROC curve
    try:
        if hasattr(model, "predict_proba"):
            y_train_proba_full = model.predict_proba(X_train)
            y_test_proba_full = model.predict_proba(X_test)
            
            # For binary classification, use positive class probability
            if y_train_proba_full.shape[1] == 2:
                y_train_proba = y_train_proba_full[:, 1]
                y_test_proba = y_test_proba_full[:, 1]
            else:
                # For multi-class, we'll use the max probability
                y_train_proba = np.max(y_train_proba_full, axis=1)
                y_test_proba = np.max(y_test_proba_full, axis=1)
        else:
            # For SVM without probability, use decision function
            y_train_proba = model.decision_function(X_train)
            y_test_proba = model.decision_function(X_test)
            
            # Handle multi-class SVM
            if len(y_train_proba.shape) > 1:
                y_train_proba = np.max(y_train_proba, axis=1)
                y_test_proba = np.max(y_test_proba, axis=1)
    except Exception as e:
        print(f"  [WARNING] Probability prediction error for {name}: {e}")
        y_train_proba = np.zeros(len(y_train))
        y_test_proba = np.zeros(len(y_test))
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # ROC AUC Score - handle multi-class case
    try:
        if len(np.unique(y)) == 2:
            # Binary classification
            train_auc = roc_auc_score(y_train, y_train_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)
        else:
            # Multi-class classification
            train_auc = roc_auc_score(y_train, y_train_proba, multi_class='ovr', average='weighted')
            test_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"  [WARNING] AUC calculation error for {name}: {e}")
        train_auc = 0.0
        test_auc = 0.0
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Store results
    results[name] = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'training_time': training_time,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_train_proba': y_train_proba,
        'y_test_proba': y_test_proba
    }
    
    print(f"  [OK] {name} trained successfully!")
    print(f"     Training time: {training_time:.2f} seconds")
    print(f"     Test Accuracy: {test_accuracy:.3f}")
    print(f"     Test F1-Score: {test_f1:.3f}")
    print(f"     Test AUC: {test_auc:.3f}")
    print(f"     CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ===================================
# STEP 5: PERFORMANCE COMPARISON
# ===================================
print("\n STEP 5: MODELS PERFORMANCE COMPARISON")
print("="*60)

# Create performance comparison DataFrame
performance_data = []
for name, metrics in results.items():
    performance_data.append({
        'Model': name,
        'Test_Accuracy': metrics['test_accuracy'],
        'Test_Precision': metrics['test_precision'],
        'Test_Recall': metrics['test_recall'],
        'Test_F1': metrics['test_f1'],
        'Test_AUC': metrics['test_auc'],
        'CV_Mean': metrics['cv_mean'],
        'CV_Std': metrics['cv_std'],
        'Training_Time': metrics['training_time']
    })

performance_df = pd.DataFrame(performance_data)
performance_df = performance_df.sort_values('Test_Accuracy', ascending=False)

print("[RANKING] MODELS PERFORMANCE RANKING:")
print(performance_df.to_string(index=False))

# Find best model
best_model_name = performance_df.iloc[0]['Model']
print(f"\n[BEST] BEST MODEL: {best_model_name}")
print(f"   Test Accuracy: {performance_df.iloc[0]['Test_Accuracy']:.3f}")
print(f"   Test F1-Score: {performance_df.iloc[0]['Test_F1']:.3f}")
print(f"   Test AUC: {performance_df.iloc[0]['Test_AUC']:.3f}")

# ===================================
# STEP 6: DETAILED CLASSIFICATION REPORTS
# ===================================
print("\n STEP 6: DETAILED CLASSIFICATION REPORTS")
print("="*60)

classification_reports = {}
for name in models.keys():
    print(f"\n{name.upper()} CLASSIFICATION REPORT:")
    print("-" * 50)
    report = classification_report(y_test, results[name]['y_test_pred'])
    print(report)
    classification_reports[name] = report

# ===================================
# STEP 7: CONFUSION MATRICES VISUALIZATION
# ===================================
print("\n STEP 7: CONFUSION MATRICES VISUALIZATION")
print("="*60)

# Plot confusion matrices for all models
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')

models_list = list(models.keys())
for i, name in enumerate(models_list):
    row = i // 2
    col = i % 2
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, results[name]['y_test_pred'])
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col])
    axes[row, col].set_title(f'{name}\nAccuracy: {results[name]["test_accuracy"]:.3f}')
    axes[row, col].set_xlabel('Predicted')
    axes[row, col].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ===================================
# STEP 8: ROC CURVES VISUALIZATION (ENHANCED)
# ===================================
print("\n STEP 8: ROC CURVES VISUALIZATION")
print("="*60)

# First, let's check the data characteristics
print(f"[INFO] Data Analysis for ROC:")
print(f"  - Unique classes in y_test: {np.unique(y_test)}")
print(f"  - Number of classes: {len(np.unique(y_test))}")
print(f"  - Class distribution: {pd.Series(y_test).value_counts().sort_index().to_dict()}")

# Check if it's binary classification
is_binary = len(np.unique(y_test)) == 2

if is_binary:
    print("  [OK] Binary classification detected - ROC curves applicable")
    
    # Plot ROC curves for all models
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    valid_curves = 0
    
    for i, (name, metrics) in enumerate(results.items()):
        try:
            y_proba = metrics['y_test_proba']
            print(f"  [PROCESS] Processing {name}:")
            print(f"     - Probabilities shape: {np.array(y_proba).shape}")
            print(f"     - Probabilities range: [{np.min(y_proba):.3f}, {np.max(y_proba):.3f}]")
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            plt.plot(fpr, tpr, color=colors[i], lw=2, 
                     label=f'{name} (AUC = {roc_auc:.3f})')
            
            print(f"     [OK] ROC AUC: {roc_auc:.3f}")
            valid_curves += 1
            
        except Exception as e:
            print(f"     [ERROR] Error: {e}")
            continue
    
    if valid_curves > 0:
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
                alpha=0.8, label='Random Classifier (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curves Comparison - Binary Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add some annotations
        plt.text(0.6, 0.2, 'Better\nClassifiers', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        plt.text(0.2, 0.8, 'Worse\nClassifiers', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        print(f"  [OK] ROC curves plotted successfully for {valid_curves} models")
    else:
        plt.close()
        print("  [ERROR] No valid ROC curves generated")

else:
    print("  [WARNING] Multi-class classification detected")
    print("  [PROCESS] Creating Multi-class ROC curves using One-vs-Rest approach...")
    
    # Multi-class ROC curves
    n_classes = len(np.unique(y_test))
    classes = sorted(np.unique(y_test))
    
    fig, axes = plt.subplots(1, min(n_classes, 3), figsize=(15, 5))
    if n_classes == 1:
        axes = [axes]
    elif n_classes == 2:
        axes = axes[:2] if hasattr(axes, '__len__') else [axes]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for class_idx, class_val in enumerate(classes[:min(3, n_classes)]):
        ax = axes[class_idx] if len(classes) > 1 else axes[0]
        
        # Create binary labels for this class vs all others
        y_binary = (y_test == class_val).astype(int)
        
        for i, (name, model) in enumerate(trained_models.items()):
            try:
                if hasattr(model, "predict_proba"):
                    y_proba_multi = model.predict_proba(X_test)
                    if class_idx < y_proba_multi.shape[1]:
                        y_proba = y_proba_multi[:, class_idx]
                    else:
                        continue
                else:
                    # Skip SVM for multi-class ROC
                    continue
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_binary, y_proba)
                roc_auc = auc(fpr, tpr)
                
                # Plot
                ax.plot(fpr, tpr, color=colors[i], lw=2,
                       label=f'{name} (AUC = {roc_auc:.3f})')
                
            except Exception as e:
                print(f"     [WARNING] Error for {name}, class {class_val}: {e}")
                continue
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - Class {class_val} vs Rest')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-class ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("  [OK] Multi-class ROC curves plotted successfully")

# Alternative: Precision-Recall Curves (useful for both binary and multi-class)
print(f"\n[ADDITIONAL] Precision-Recall Curves")
print("-" * 40)

plt.figure(figsize=(12, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple']

for i, (name, metrics) in enumerate(results.items()):
    try:
        if is_binary:
            # Binary case
            precision, recall, _ = precision_recall_curve(y_test, metrics['y_test_proba'])
            avg_precision = np.trapz(precision, recall)
            
            plt.plot(recall, precision, color=colors[i], lw=2,
                    label=f'{name} (AP = {avg_precision:.3f})')
        else:
            # Multi-class case - use macro average
            y_test_binary = label_binarize(y_test, classes=classes)
            if hasattr(trained_models[name], "predict_proba"):
                y_proba_multi = trained_models[name].predict_proba(X_test)
                
                precisions = []
                recalls = []
                for j in range(len(classes)):
                    if j < y_proba_multi.shape[1]:
                        precision, recall, _ = precision_recall_curve(
                            y_test_binary[:, j], y_proba_multi[:, j])
                        precisions.append(precision)
                        recalls.append(recall)
                
                if precisions:
                    # Use the first class curve as representative
                    plt.plot(recalls[0], precisions[0], color=colors[i], lw=2,
                            label=f'{name} (Class 0)')
            
    except Exception as e:
        print(f"  [WARNING] PR curve error for {name}: {e}")
        continue

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("  [OK] Precision-Recall curves plotted successfully")

# ===================================
# STEP 9: PERFORMANCE METRICS VISUALIZATION
# ===================================
print("\n STEP 9: PERFORMANCE METRICS VISUALIZATION")
print("="*60)

# Create comprehensive performance visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Models Performance Metrics Comparison', fontsize=16, fontweight='bold')

# 1. Accuracy Comparison
models_names = performance_df['Model']
test_accuracies = performance_df['Test_Accuracy']
cv_accuracies = performance_df['CV_Mean']

x_pos = np.arange(len(models_names))
width = 0.35

ax1.bar(x_pos - width/2, test_accuracies, width, label='Test Accuracy', color='skyblue')
ax1.bar(x_pos + width/2, cv_accuracies, width, label='CV Accuracy', color='lightcoral')
ax1.set_xlabel('Models')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for i, (test_acc, cv_acc) in enumerate(zip(test_accuracies, cv_accuracies)):
    ax1.text(i - width/2, test_acc + 0.01, f'{test_acc:.3f}', ha='center', va='bottom')
    ax1.text(i + width/2, cv_acc + 0.01, f'{cv_acc:.3f}', ha='center', va='bottom')

# 2. All Metrics Comparison
metrics_to_plot = ['Test_Precision', 'Test_Recall', 'Test_F1', 'Test_AUC']
metric_colors = ['lightgreen', 'lightyellow', 'lightpink', 'lightcyan']

for i, metric in enumerate(metrics_to_plot):
    values = performance_df[metric]
    ax2.bar(x_pos + i*0.2 - 0.3, values, 0.2, label=metric.replace('Test_', ''), 
           color=metric_colors[i])

ax2.set_xlabel('Models')
ax2.set_ylabel('Score')
ax2.set_title('All Metrics Comparison')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models_names, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Training Time Comparison
training_times_list = performance_df['Training_Time']
bars = ax3.bar(models_names, training_times_list, color='lightsteelblue')
ax3.set_xlabel('Models')
ax3.set_ylabel('Training Time (seconds)')
ax3.set_title('Training Time Comparison')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# Add value labels on bars
for i, time_val in enumerate(training_times_list):
    ax3.text(i, time_val + 0.01, f'{time_val:.2f}s', ha='center', va='bottom')

# 4. Model Performance Radar Chart (simplified bar version)
models_for_radar = models_names[:4]  # Top 4 models
metrics_for_radar = performance_df[['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']].head(4)

x_radar = np.arange(len(metrics_for_radar.columns))
width_radar = 0.2

for i, (idx, row) in enumerate(metrics_for_radar.iterrows()):
    model_name = performance_df.iloc[idx]['Model']
    ax4.bar(x_radar + i*width_radar, row.values, width_radar, label=model_name)

ax4.set_xlabel('Metrics')
ax4.set_ylabel('Score')
ax4.set_title('Top 4 Models - Key Metrics')
ax4.set_xticks(x_radar + width_radar * 1.5)
ax4.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===================================
# STEP 10: FEATURE IMPORTANCE (FOR TREE-BASED MODELS)
# ===================================
print("\n STEP 10: FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Extract feature importance for tree-based models
tree_based_models = ['Decision Tree', 'Random Forest']
importances_data = {}

for model_name in tree_based_models:
    if model_name in trained_models:
        model = trained_models[model_name]
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X_selected.columns
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            importances_data[model_name] = importance_df
            
            print(f"\n[TOP 10] {model_name} - Top 10 Feature Importances:")
            print(importance_df.head(10).to_string(index=False))

# Visualize feature importances
if importances_data:
    fig, axes = plt.subplots(1, len(importances_data), figsize=(16, 6))
    if len(importances_data) == 1:
        axes = [axes]
    
    for i, (model_name, importance_df) in enumerate(importances_data.items()):
        top_features = importance_df.head(10)
        
        axes[i].barh(range(len(top_features)), top_features['importance'], color='skyblue')
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels(top_features['feature'])
        axes[i].set_xlabel('Importance')
        axes[i].set_title(f'{model_name} - Feature Importance')
        axes[i].invert_yaxis()
    
    plt.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ===================================
# STEP 11: SAVE MODELS AND RESULTS
# ===================================
print("\n STEP 11: SAVING MODELS AND RESULTS")
print("="*60)

# Add this to complete the STEP 11 section after line 674

try:
    # Create directories if they don't exist
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save trained models
    for name, model in trained_models.items():
        model_filename = f'models/{name.lower().replace(" ", "_")}_model.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"  [OK] {name} model saved to: {model_filename}")
    
    # Save best model separately
    best_model = trained_models[best_model_name]
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"  [BEST] Best model ({best_model_name}) saved to: models/best_model.pkl")
    
    # Save performance results
    performance_df.to_csv('results/models_performance_comparison.csv', index=False)
    print(f"  [OK] Performance comparison saved to: results/models_performance_comparison.csv")
    
    # Save detailed results with proper encoding
    with open('results/classification_reports.txt', 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATION REPORTS FOR ALL MODELS\n")
        f.write("="*60 + "\n\n")
        for name, report in classification_reports.items():
            f.write(f"{name.upper()}:\n")
            f.write("-" * 40 + "\n")
            f.write(report)
            f.write("\n" + "="*60 + "\n\n")
    print(f"  [OK] Classification reports saved to: results/classification_reports.txt")
    
    # Save feature importance data
    if importances_data:
        for model_name, importance_df in importances_data.items():
            filename = f'results/{model_name.lower().replace(" ", "_")}_feature_importance.csv'
            importance_df.to_csv(filename, index=False)
            print(f"  [OK] {model_name} feature importance saved to: {filename}")
    
    # Create a summary report
    summary_report = f"""
SUPERVISED LEARNING MODELS - SUMMARY REPORT
==========================================

BEST MODEL: {best_model_name}
Test Accuracy: {performance_df.iloc[0]['Test_Accuracy']:.3f}
Test F1-Score: {performance_df.iloc[0]['Test_F1']:.3f}
Test AUC: {performance_df.iloc[0]['Test_AUC']:.3f}
Training Time: {performance_df.iloc[0]['Training_Time']:.2f} seconds

MODELS RANKING (by Test Accuracy):
"""
    
    for i, row in performance_df.iterrows():
        summary_report += f"{i+1}. {row['Model']}: {row['Test_Accuracy']:.3f}\n"
    
    summary_report += f"""

DATA INFORMATION:
- Training samples: {X_train.shape[0]}
- Testing samples: {X_test.shape[0]}
- Features used: {X_train.shape[1]}
- Target classes: {len(np.unique(y_test))}

FILES GENERATED:
- Model files: models/[model_name]_model.pkl
- Best model: models/best_model.pkl
- Performance comparison: results/models_performance_comparison.csv
- Classification reports: results/classification_reports.txt
- Feature importance: results/[model]_feature_importance.csv

READY FOR NEXT PHASE: Hyperparameter Tuning
"""
    
    with open('results/supervised_learning_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_report)
    print(f"  [OK] Summary report saved to: results/supervised_learning_summary.txt")
    
    print(f"\n[SUCCESS] All models and results saved successfully!")
    
except Exception as e:
    print(f"[ERROR] Error saving models and results: {e}")
    print("[INFO] Some files may not have been saved properly")

# ===================================
# STEP 12: FINAL SUMMARY AND RECOMMENDATIONS
# ===================================
print("\n" + "="*80)
print("SUPERVISED LEARNING - FINAL SUMMARY")
print("="*80)

print(f"\n[BEST MODEL] {best_model_name}")
print(f"  Test Accuracy: {performance_df.iloc[0]['Test_Accuracy']:.3f}")
print(f"  Test Precision: {performance_df.iloc[0]['Test_Precision']:.3f}")
print(f"  Test Recall: {performance_df.iloc[0]['Test_Recall']:.3f}")
print(f"  Test F1-Score: {performance_df.iloc[0]['Test_F1']:.3f}")
print(f"  Test AUC: {performance_df.iloc[0]['Test_AUC']:.3f}")
print(f"  CV Accuracy: {performance_df.iloc[0]['CV_Mean']:.3f} ± {performance_df.iloc[0]['CV_Std']:.3f}")
print(f"  Training Time: {performance_df.iloc[0]['Training_Time']:.2f} seconds")

print(f"\n[MODELS PERFORMANCE RANKING]:")
for i, row in performance_df.iterrows():
    print(f"  {i+1}. {row['Model']}: {row['Test_Accuracy']:.3f} accuracy")

print(f"\n[KEY INSIGHTS]:")
print(f"  - Dataset size: {len(X_selected)} samples with {len(X_selected.columns)} selected features")
print(f"  - Classification type: {'Binary' if len(np.unique(y)) == 2 else 'Multi-class'} classification")
print(f"  - Best performing algorithm: {best_model_name}")
if importances_data and best_model_name in importances_data:
    top_feature = importances_data[best_model_name].iloc[0]['feature']
    print(f"  - Most important feature (if tree-based): {top_feature}")

print(f"\n[RECOMMENDATIONS]:")
if performance_df.iloc[0]['Test_Accuracy'] > 0.85:
    print("  - Excellent model performance achieved!")
elif performance_df.iloc[0]['Test_Accuracy'] > 0.80:
    print("  - Good model performance achieved!")
else:
    print("  - Model performance needs improvement. Consider:")
    print("    * More feature engineering")
    print("    * Different algorithms")
    print("    * More data collection")

print("  - Next steps: Hyperparameter tuning for further optimization")
print("  - Consider ensemble methods for improved performance")

print(f"\n[FILES CREATED]:")
print("  - models/best_model.pkl (ready for deployment)")
print("  - results/models_performance_comparison.csv")
print("  - results/supervised_learning_summary.txt")

print(f"\n" + "="*80)
print("SUPERVISED LEARNING COMPLETED SUCCESSFULLY!")
print("Ready for hyperparameter tuning phase...")
print("="*80)
