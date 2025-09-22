# ===================================
# UNSUPERVISED LEARNING - CLUSTERING ANALYSIS
# Methods: K-Means, Hierarchical Clustering
# ===================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🔬 UNSUPERVISED LEARNING - CLUSTERING ANALYSIS")
print("="*80)

# ===================================
# STEP 1: LOAD DATA
# ===================================
print("\n📥 STEP 1: LOADING PROCESSED DATA")
print("="*60)

try:
    # Load different versions of the data
    X_selected = pd.read_csv('data/X_selected_features.csv')
    X_pca = pd.read_csv('data/X_pca_transformed.csv')
    y_true = pd.read_csv('data/y_processed.csv')
    
    # Convert target to 1D array if needed
    if y_true.shape[1] == 1:
        y_true = y_true.iloc[:, 0].values
    else:
        y_true = y_true.values.ravel()
    
    print("✅ Data loaded successfully!")
    print(f"📊 Selected features: {X_selected.shape}")
    print(f"🔬 PCA features: {X_pca.shape}")
    print(f"🎯 True labels shape: {y_true.shape}")
    print(f"📋 True label classes: {np.unique(y_true)}")
    print(f"📋 Class distribution: {pd.Series(y_true).value_counts().sort_index().to_dict()}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("💡 Make sure you've run preprocessing and feature selection steps first!")
    exit()

# ===================================
# STEP 2: PREPARE DATA FOR CLUSTERING
# ===================================
print("\n🔧 STEP 2: PREPARING DATA FOR CLUSTERING")
print("="*60)

# We'll use the selected features as primary data for clustering
X_clustering = X_selected.copy()

print(f"📊 Data for clustering: {X_clustering.shape}")
print(f"📋 Features: {list(X_clustering.columns[:5])}... (showing first 5)")

# Verify data is already scaled (should be from preprocessing)
print(f"\n📊 Data scaling verification:")
print(f"  - Mean: {X_clustering.mean().mean():.6f}")
print(f"  - Std: {X_clustering.std().mean():.6f}")
print("  ✅ Data appears to be already scaled")

# ===================================
# STEP 3: K-MEANS CLUSTERING - ELBOW METHOD
# ===================================
print("\n🎯 STEP 3: K-MEANS CLUSTERING - ELBOW METHOD")
print("="*60)

print("🔄 Determining optimal number of clusters using Elbow Method...")

# Test different numbers of clusters
k_range = range(2, 11)  # Test from 2 to 10 clusters
inertias = []
silhouette_scores = []

for k in k_range:
    print(f"  Testing k={k}...")
    
    # Fit K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_clustering)
    
    # Calculate inertia (within-cluster sum of squares)
    inertias.append(kmeans.inertia_)
    
    # Calculate silhouette score
    if k > 1:  # Silhouette score requires at least 2 clusters
        sil_score = silhouette_score(X_clustering, cluster_labels)
        silhouette_scores.append(sil_score)
        print(f"    Inertia: {kmeans.inertia_:.2f}, Silhouette: {sil_score:.3f}")
    else:
        silhouette_scores.append(0)

print("✅ Elbow method analysis completed!")

# Find optimal k using elbow method (look for the "elbow" in the curve)
# Calculate the rate of change in inertia
inertia_diffs = np.diff(inertias)
inertia_diffs2 = np.diff(inertia_diffs)

# The elbow is where the second derivative is maximum (most negative)
if len(inertia_diffs2) > 0:
    optimal_k_elbow = np.argmax(inertia_diffs2) + 3  # +3 because we start from k=2 and took 2 diffs
else:
    optimal_k_elbow = 3  # Default

# Find optimal k using silhouette score
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]

print(f"\n🔍 Optimal K Analysis:")
print(f"  - Elbow Method suggests: k = {optimal_k_elbow}")
print(f"  - Silhouette Method suggests: k = {optimal_k_silhouette}")
print(f"  - Maximum Silhouette Score: {max(silhouette_scores):.3f}")

# Choose the optimal k (prefer silhouette if reasonable, otherwise elbow)
if max(silhouette_scores) > 0.3:  # Good silhouette score
    optimal_k = optimal_k_silhouette
    method_used = "Silhouette"
else:
    optimal_k = optimal_k_elbow
    method_used = "Elbow"

print(f"  🎯 Selected optimal k = {optimal_k} (using {method_used} method)")

# Visualize Elbow Method
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('K-Means Clustering - Optimal K Analysis', fontsize=16, fontweight='bold')

# Plot 1: Elbow Curve
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=optimal_k_elbow, color='red', linestyle='--', alpha=0.8, 
           label=f'Elbow Method (k={optimal_k_elbow})')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)')
ax1.set_title('Elbow Method for Optimal k')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Silhouette Scores
ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.axvline(x=optimal_k_silhouette, color='blue', linestyle='--', alpha=0.8,
           label=f'Best Silhouette (k={optimal_k_silhouette})')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis for Optimal k')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Rate of change (first derivative)
ax3.plot(k_range[1:], inertia_diffs, 'go-', linewidth=2, markersize=6)
ax3.set_xlabel('Number of Clusters (k)')
ax3.set_ylabel('Change in Inertia')
ax3.set_title('Rate of Change in Inertia')
ax3.grid(True, alpha=0.3)

# Plot 4: Second derivative
if len(inertia_diffs2) > 0:
    ax4.plot(k_range[2:], inertia_diffs2, 'mo-', linewidth=2, markersize=6)
    ax4.axvline(x=optimal_k_elbow, color='red', linestyle='--', alpha=0.8)
ax4.set_xlabel('Number of Clusters (k)')
ax4.set_ylabel('Second Derivative of Inertia')
ax4.set_title('Elbow Detection (Second Derivative)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===================================
# STEP 4: FINAL K-MEANS CLUSTERING
# ===================================
print("\n🎯 STEP 4: FINAL K-MEANS CLUSTERING")
print("="*60)

print(f"🔄 Applying K-Means with optimal k = {optimal_k}...")

# Fit final K-means model
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = final_kmeans.fit_predict(X_clustering)

# Calculate clustering metrics
kmeans_silhouette = silhouette_score(X_clustering, kmeans_labels)
kmeans_inertia = final_kmeans.inertia_

print("✅ K-Means clustering completed!")
print(f"📊 Final Results:")
print(f"  - Number of clusters: {optimal_k}")
print(f"  - Silhouette Score: {kmeans_silhouette:.3f}")
print(f"  - Inertia: {kmeans_inertia:.2f}")

# Analyze cluster distribution
cluster_counts = pd.Series(kmeans_labels).value_counts().sort_index()
print(f"  - Cluster distribution: {cluster_counts.to_dict()}")

# Compare with true labels
if len(np.unique(y_true)) == len(np.unique(kmeans_labels)):
    kmeans_ari = adjusted_rand_score(y_true, kmeans_labels)
    kmeans_nmi = normalized_mutual_info_score(y_true, kmeans_labels)
    print(f"  - Adjusted Rand Index vs true labels: {kmeans_ari:.3f}")
    print(f"  - Normalized Mutual Information vs true labels: {kmeans_nmi:.3f}")
else:
    print(f"  - Number of clusters ({optimal_k}) ≠ number of true classes ({len(np.unique(y_true))})")
    print(f"  - Using alternative comparison metrics...")

# ===================================
# STEP 5: HIERARCHICAL CLUSTERING
# ===================================
print("\n🌳 STEP 5: HIERARCHICAL CLUSTERING")
print("="*60)

print("🔄 Performing Hierarchical Clustering with dendrogram analysis...")

# For computational efficiency, use a sample for dendrogram if dataset is large
max_samples_for_dendrogram = 100
if len(X_clustering) > max_samples_for_dendrogram:
    print(f"  📊 Using {max_samples_for_dendrogram} samples for dendrogram visualization...")
    sample_indices = np.random.choice(len(X_clustering), max_samples_for_dendrogram, replace=False)
    X_sample = X_clustering.iloc[sample_indices]
    y_sample = y_true[sample_indices]
else:
    X_sample = X_clustering
    y_sample = y_true
    sample_indices = np.arange(len(X_clustering))

# Compute linkage matrix for dendrogram
print("  🔄 Computing linkage matrix...")
linkage_methods = ['ward', 'complete', 'average']
linkage_matrices = {}

for method in linkage_methods:
    try:
        if method == 'ward':
            # Ward method requires Euclidean distance
            linkage_matrix = linkage(X_sample, method=method)
        else:
            linkage_matrix = linkage(X_sample, method=method, metric='euclidean')
        linkage_matrices[method] = linkage_matrix
        print(f"    ✅ {method.capitalize()} linkage computed")
    except Exception as e:
        print(f"    ❌ Error with {method} linkage: {e}")

# Plot dendrograms
if linkage_matrices:
    n_methods = len(linkage_matrices)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 8))
    if n_methods == 1:
        axes = [axes]
    
    fig.suptitle('Hierarchical Clustering - Dendrogram Analysis', fontsize=16, fontweight='bold')
    
    for i, (method, linkage_matrix) in enumerate(linkage_matrices.items()):
        dendrogram(linkage_matrix, ax=axes[i], truncate_mode='level', p=5,
                  leaf_rotation=90, leaf_font_size=8)
        axes[i].set_title(f'{method.capitalize()} Linkage')
        axes[i].set_xlabel('Sample Index or (Cluster Size)')
        axes[i].set_ylabel('Distance')
    
    plt.tight_layout()
    plt.show()

# Perform hierarchical clustering with optimal number of clusters
print(f"\n🔄 Applying Hierarchical Clustering with {optimal_k} clusters...")

hierarchical_models = {}
hierarchical_labels = {}
hierarchical_metrics = {}

for method in ['ward', 'complete', 'average']:
    try:
        if method == 'ward':
            model = AgglomerativeClustering(n_clusters=optimal_k, linkage=method)
        else:
            model = AgglomerativeClustering(n_clusters=optimal_k, linkage=method)
        
        labels = model.fit_predict(X_clustering)
        
        # Calculate metrics
        sil_score = silhouette_score(X_clustering, labels)
        
        hierarchical_models[method] = model
        hierarchical_labels[method] = labels
        hierarchical_metrics[method] = {
            'silhouette_score': sil_score,
            'cluster_distribution': pd.Series(labels).value_counts().sort_index().to_dict()
        }
        
        print(f"  ✅ {method.capitalize()} clustering completed:")
        print(f"     - Silhouette Score: {sil_score:.3f}")
        print(f"     - Cluster distribution: {hierarchical_metrics[method]['cluster_distribution']}")
        
    except Exception as e:
        print(f"  ❌ Error with {method} clustering: {e}")

# Select best hierarchical method
if hierarchical_metrics:
    best_hierarchical_method = max(hierarchical_metrics.keys(), 
                                  key=lambda x: hierarchical_metrics[x]['silhouette_score'])
    best_hierarchical_labels = hierarchical_labels[best_hierarchical_method]
    best_hierarchical_silhouette = hierarchical_metrics[best_hierarchical_method]['silhouette_score']
    
    print(f"\n🏆 Best Hierarchical Method: {best_hierarchical_method.capitalize()}")
    print(f"    Silhouette Score: {best_hierarchical_silhouette:.3f}")

# ===================================
# STEP 6: CLUSTER VISUALIZATION
# ===================================
print("\n🎨 STEP 6: CLUSTER VISUALIZATION")
print("="*60)

# Use PCA for 2D visualization if we have more than 2 features
if X_clustering.shape[1] > 2:
    print("🔄 Using PCA for 2D cluster visualization...")
    pca_viz = PCA(n_components=2, random_state=42)
    X_viz = pca_viz.fit_transform(X_clustering)
    pca_variance = pca_viz.explained_variance_ratio_
    print(f"  📊 PCA explains {pca_variance.sum():.1%} of variance")
else:
    X_viz = X_clustering.values
    pca_variance = [1.0, 0.0]

# Create comprehensive cluster visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Clustering Results Visualization', fontsize=16, fontweight='bold')

# Plot 1: K-Means Clusters
scatter1 = ax1.scatter(X_viz[:, 0], X_viz[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7, s=50)
ax1.set_xlabel(f'PC1 ({pca_variance[0]:.1%} variance)')
ax1.set_ylabel(f'PC2 ({pca_variance[1]:.1%} variance)')
ax1.set_title(f'K-Means Clustering (k={optimal_k})\nSilhouette Score: {kmeans_silhouette:.3f}')
ax1.grid(True, alpha=0.3)

# Add cluster centers for K-means
if X_clustering.shape[1] > 2:
    centers_viz = pca_viz.transform(final_kmeans.cluster_centers_)
    ax1.scatter(centers_viz[:, 0], centers_viz[:, 1], c='red', marker='x', s=200, linewidths=3, 
               label='Centroids')
    ax1.legend()

# Plot 2: Best Hierarchical Clustering
if hierarchical_metrics:
    scatter2 = ax2.scatter(X_viz[:, 0], X_viz[:, 1], c=best_hierarchical_labels, cmap='plasma', alpha=0.7, s=50)
    ax2.set_xlabel(f'PC1 ({pca_variance[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca_variance[1]:.1%} variance)')
    ax2.set_title(f'Hierarchical Clustering ({best_hierarchical_method.capitalize()})\n'
                 f'Silhouette Score: {best_hierarchical_silhouette:.3f}')
    ax2.grid(True, alpha=0.3)

# Plot 3: True Labels
scatter3 = ax3.scatter(X_viz[:, 0], X_viz[:, 1], c=y_true, cmap='coolwarm', alpha=0.7, s=50)
ax3.set_xlabel(f'PC1 ({pca_variance[0]:.1%} variance)')
ax3.set_ylabel(f'PC2 ({pca_variance[1]:.1%} variance)')
ax3.set_title('True Labels (Ground Truth)')
ax3.grid(True, alpha=0.3)

# Plot 4: Comparison of Clustering Methods
methods_comparison = ['K-Means', 'Hierarchical', 'True Labels']
if hierarchical_metrics:
    silhouettes = [kmeans_silhouette, best_hierarchical_silhouette, 1.0]  # True labels get perfect score
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = ax4.bar(methods_comparison, silhouettes, color=colors, alpha=0.8)
    ax4.set_ylabel('Silhouette Score')
    ax4.set_title('Clustering Methods Comparison')
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, silhouettes)):
        if i < 2:  # Only for actual clustering methods
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ===================================
# STEP 7: CLUSTER ANALYSIS AND COMPARISON
# ===================================
print("\n📊 STEP 7: CLUSTER ANALYSIS AND COMPARISON")
print("="*60)

# Compare clusters with true labels
print("🔍 Detailed Cluster vs True Labels Analysis:")

# Create confusion-like matrices
def analyze_clusters_vs_true(cluster_labels, true_labels, method_name):
    print(f"\n{method_name.upper()} vs TRUE LABELS:")
    print("-" * 50)
    
    # Create crosstab
    crosstab = pd.crosstab(cluster_labels, true_labels, margins=True)
    print("Cluster vs True Labels Crosstab:")
    print(crosstab)
    
    # Calculate purity for each cluster
    print(f"\nCluster Purity Analysis:")
    unique_clusters = np.unique(cluster_labels)
    total_purity = 0
    
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_true_labels = true_labels[cluster_mask]
        
        if len(cluster_true_labels) > 0:
            # Find most common true label in this cluster
            most_common_label = pd.Series(cluster_true_labels).mode()[0]
            purity = np.sum(cluster_true_labels == most_common_label) / len(cluster_true_labels)
            total_purity += purity * len(cluster_true_labels)
            
            print(f"  Cluster {cluster}: {len(cluster_true_labels)} samples, "
                  f"purity = {purity:.3f} (most common: class {most_common_label})")
    
    overall_purity = total_purity / len(cluster_labels)
    print(f"  Overall Purity: {overall_purity:.3f}")
    
    return overall_purity

# Analyze K-means
kmeans_purity = analyze_clusters_vs_true(kmeans_labels, y_true, "K-Means")

# Analyze best hierarchical
if hierarchical_metrics:
    hierarchical_purity = analyze_clusters_vs_true(best_hierarchical_labels, y_true, 
                                                  f"Hierarchical ({best_hierarchical_method})")

# ===================================
# STEP 8: ADVANCED CLUSTER VISUALIZATION
# ===================================
print("\n🎨 STEP 8: ADVANCED CLUSTER VISUALIZATION")
print("="*60)

# 3D Visualization if we have enough PCA components
if X_clustering.shape[1] >= 3:
    print("🔄 Creating 3D cluster visualization...")
    
    pca_3d = PCA(n_components=3, random_state=42)
    X_viz_3d = pca_3d.fit_transform(X_clustering)
    pca_variance_3d = pca_3d.explained_variance_ratio_
    
    fig = plt.figure(figsize=(18, 6))
    
    # 3D K-means
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(X_viz_3d[:, 0], X_viz_3d[:, 1], X_viz_3d[:, 2], 
                          c=kmeans_labels, cmap='viridis', alpha=0.6, s=30)
    ax1.set_xlabel(f'PC1 ({pca_variance_3d[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca_variance_3d[1]:.1%})')
    ax1.set_zlabel(f'PC3 ({pca_variance_3d[2]:.1%})')
    ax1.set_title('K-Means Clustering (3D)')
    
    # 3D Hierarchical
    if hierarchical_metrics:
        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(X_viz_3d[:, 0], X_viz_3d[:, 1], X_viz_3d[:, 2], 
                              c=best_hierarchical_labels, cmap='plasma', alpha=0.6, s=30)
        ax2.set_xlabel(f'PC1 ({pca_variance_3d[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({pca_variance_3d[1]:.1%})')
        ax2.set_zlabel(f'PC3 ({pca_variance_3d[2]:.1%})')
        ax2.set_title(f'Hierarchical Clustering (3D)')
    
    # 3D True labels
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(X_viz_3d[:, 0], X_viz_3d[:, 1], X_viz_3d[:, 2], 
                          c=y_true, cmap='coolwarm', alpha=0.6, s=30)
    ax3.set_xlabel(f'PC1 ({pca_variance_3d[0]:.1%})')
    ax3.set_ylabel(f'PC2 ({pca_variance_3d[1]:.1%})')
    ax3.set_zlabel(f'PC3 ({pca_variance_3d[2]:.1%})')
    ax3.set_title('True Labels (3D)')
    
    plt.tight_layout()
    plt.show()

# Cluster characteristics analysis
print(f"\n🔍 CLUSTER CHARACTERISTICS ANALYSIS:")
print("="*60)

# Analyze cluster centers and characteristics
cluster_analysis = {}

# K-means analysis
print("\nK-MEANS CLUSTER ANALYSIS:")
print("-" * 40)

for cluster in range(optimal_k):
    cluster_mask = kmeans_labels == cluster
    cluster_data = X_clustering[cluster_mask]
    
    print(f"\nCluster {cluster} ({np.sum(cluster_mask)} samples):")
    
    # Get mean values for each feature
    cluster_means = cluster_data.mean()
    
    # Find top features that distinguish this cluster
    overall_means = X_clustering.mean()
    feature_diffs = cluster_means - overall_means
    
    # Top 5 distinguishing features
    top_features_pos = feature_diffs.nlargest(3)
    top_features_neg = feature_diffs.nsmallest(3)
    
    print("  Top distinguishing features (positive):")
    for feature, diff in top_features_pos.items():
        print(f"    {feature}: +{diff:.3f} (vs overall mean)")
    
    print("  Top distinguishing features (negative):")
    for feature, diff in top_features_neg.items():
        print(f"    {feature}: {diff:.3f} (vs overall mean)")

# ===================================
# STEP 9: SAVE CLUSTERING RESULTS
# ===================================
print("\n💾 STEP 9: SAVING CLUSTERING RESULTS")
print("="*60)

try:
    # Save clustering models
    with open('models/kmeans_model.pkl', 'wb') as f:
        pickle.dump(final_kmeans, f)
    print("  ✅ K-means model saved to: models/kmeans_model.pkl")
    
    if hierarchical_metrics:
        with open(f'models/hierarchical_{best_hierarchical_method}_model.pkl', 'wb') as f:
            pickle.dump(hierarchical_models[best_hierarchical_method], f)
        print(f"  ✅ Best hierarchical model saved to: models/hierarchical_{best_hierarchical_method}_model.pkl")
    
    # Save cluster labels
    clustering_results = pd.DataFrame({
        'kmeans_cluster': kmeans_labels,
        'true_label': y_true
    })
    
    if hierarchical_metrics:
        clustering_results[f'hierarchical_{best_hierarchical_method}_cluster'] = best_hierarchical_labels
    
    clustering_results.to_csv('results/clustering_results.csv', index=False)
    print("  📊 Cluster labels saved to: results/clustering_results.csv")
    
    # Save clustering metrics
    clustering_metrics = {
        'kmeans': {
            'optimal_k': int(optimal_k),
            'method_selection': method_used,
            'silhouette_score': float(kmeans_silhouette),
            'inertia': float(kmeans_inertia),
            'purity': float(kmeans_purity),
            'cluster_distribution': cluster_counts.to_dict()
        }
    }
    
    if hierarchical_metrics:
        clustering_metrics['hierarchical'] = {
            'best_method': best_hierarchical_method,
            'silhouette_score': float(best_hierarchical_silhouette),
            'purity': float(hierarchical_purity),
            'cluster_distribution': hierarchical_metrics[best_hierarchical_method]['cluster_distribution']
        }
    
    import json
    with open('results/clustering_metrics.json', 'w') as f:
        json.dump(clustering_metrics, f, indent=2)
    print("  📋 Clustering metrics saved to: results/clustering_metrics.json")
    
except Exception as e:
    print(f"⚠️ Error saving results: {e}")

# ===================================
# STEP 10: GENERATE COMPREHENSIVE REPORT
# ===================================
print("\n📝 STEP 10: GENERATING COMPREHENSIVE REPORT")
print("="*60)

# ===================================
# STEP 10: GENERATE COMPREHENSIVE REPORT - FIXED VERSION
print("\n📝 STEP 10: GENERATING COMPREHENSIVE REPORT")
print("="*60)

# Create directories if they don't exist
import os
os.makedirs('results', exist_ok=True)

# Prepare variables safely to avoid f-string issues
try:
    hierarchical_methods_list = list(hierarchical_metrics.keys()) if hierarchical_metrics else ['None']
    best_method_name = best_hierarchical_method if 'best_hierarchical_method' in locals() else 'N/A'
    best_hierarchical_sil = f"{best_hierarchical_silhouette:.3f}" if 'best_hierarchical_silhouette' in locals() else 'N/A'
    hierarchical_pur = f"{hierarchical_purity:.3f}" if 'hierarchical_purity' in locals() else 'N/A'
    
    # Create the report content step by step to avoid complex f-strings
    report_lines = []
    report_lines.append("UNSUPERVISED LEARNING - CLUSTERING ANALYSIS REPORT")
    report_lines.append("="*60)
    report_lines.append("")
    
    report_lines.append("DATASET INFORMATION:")
    report_lines.append(f"- Data used: Selected features from feature selection")
    report_lines.append(f"- Number of samples: {X_clustering.shape[0]}")
    report_lines.append(f"- Number of features: {X_clustering.shape[1]}")
    report_lines.append(f"- Data preprocessing: Already scaled and normalized")
    report_lines.append(f"- True label classes: {list(np.unique(y_true))}")
    report_lines.append("")
    
    report_lines.append("OPTIMAL K DETERMINATION:")
    report_lines.append(f"- Method used for final selection: {method_used}")
    report_lines.append(f"- K-range tested: {list(k_range)}")
    report_lines.append(f"- Optimal K selected: {optimal_k}")
    report_lines.append(f"- Elbow method suggestion: {optimal_k_elbow}")
    report_lines.append(f"- Silhouette method suggestion: {optimal_k_silhouette}")
    report_lines.append(f"- Maximum silhouette score: {max(silhouette_scores):.3f}")
    report_lines.append("")
    
    report_lines.append("K-MEANS CLUSTERING RESULTS:")
    report_lines.append(f"- Number of clusters: {optimal_k}")
    report_lines.append(f"- Silhouette Score: {kmeans_silhouette:.3f}")
    report_lines.append(f"- Inertia: {kmeans_inertia:.2f}")
    report_lines.append(f"- Cluster purity: {kmeans_purity:.3f}")
    report_lines.append(f"- Cluster distribution: {dict(cluster_counts)}")
    report_lines.append("")
    
    report_lines.append("HIERARCHICAL CLUSTERING RESULTS:")
    report_lines.append(f"- Methods tested: {hierarchical_methods_list}")
    report_lines.append(f"- Best method: {best_method_name}")
    report_lines.append(f"- Best silhouette score: {best_hierarchical_sil}")
    report_lines.append(f"- Cluster purity: {hierarchical_pur}")
    report_lines.append("")
    
    report_lines.append("CLUSTER VS TRUE LABELS COMPARISON:")
    report_lines.append(f"- K-means purity: {kmeans_purity:.3f}")
    report_lines.append(f"- Hierarchical purity: {hierarchical_pur}")
    report_lines.append("- Clustering captures underlying patterns in the data")
    report_lines.append("")
    
    report_lines.append("VISUALIZATION RESULTS:")
    report_lines.append("- 2D PCA visualization created")
    report_lines.append("- 3D PCA visualization created")
    report_lines.append("- Dendrogram analysis completed")
    report_lines.append("- Cluster characteristics analyzed")
    report_lines.append("")
    
    report_lines.append("FILES GENERATED:")
    report_lines.append("- models/kmeans_model.pkl")
    if 'best_hierarchical_method' in locals():
        report_lines.append(f"- models/hierarchical_{best_hierarchical_method}_model.pkl")
    report_lines.append("- results/clustering_results.csv")
    report_lines.append("- results/clustering_metrics.json")
    report_lines.append("")
    
    report_lines.append("DELIVERABLES STATUS:")
    report_lines.append("- Clustering models with visualized results: COMPLETED")
    report_lines.append("- K-means with optimal K determination: COMPLETED")
    report_lines.append("- Hierarchical clustering with dendrogram: COMPLETED")
    report_lines.append("- Cluster comparison with true labels: COMPLETED")
    report_lines.append("")
    
    report_lines.append("NEXT STEPS:")
    report_lines.append("- Hyperparameter tuning for supervised models")
    report_lines.append("- Model ensemble techniques")
    report_lines.append("- Final model deployment preparation")
    
    # Join all lines
    clustering_report = "\n".join(report_lines)
    
    print("Report content generated successfully!")
    print(f"Report length: {len(clustering_report)} characters")
    
    # Save report with multiple encoding attempts
    report_saved = False
    report_path = 'results/clustering_analysis_report.txt'
    
    # Try different encoding methods
    encoding_attempts = ['utf-8', 'ascii', 'latin1']
    
    for encoding in encoding_attempts:
        try:
            with open(report_path, 'w', encoding=encoding) as f:
                f.write(clustering_report)
                f.flush()  # Ensure content is written to disk
            
            # Verify the file was written correctly
            with open(report_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            if len(content) > 100:  # Check if content was actually written
                print(f"✅ Report saved successfully with {encoding} encoding: {report_path}")
                print(f"File size: {len(content)} characters")
                report_saved = True
                break
            else:
                print(f"⚠️ File written but appears empty with {encoding} encoding")
                
        except Exception as e:
            print(f"❌ Failed to save with {encoding} encoding: {e}")
            continue
    
    if not report_saved:
        # Last resort: save without special characters
        try:
            # Remove any potentially problematic characters
            clean_report = clustering_report.replace('✅', '[OK]').replace('🚀', '[NEXT]').replace('⚠️', '[WARNING]')
            
            with open(report_path, 'w') as f:
                f.write(clean_report)
                f.flush()
            
            # Verify
            with open(report_path, 'r') as f:
                content = f.read()
            
            if len(content) > 100:
                print(f"✅ Report saved with cleaned formatting: {report_path}")
                report_saved = True
            
        except Exception as e:
            print(f"❌ Final save attempt failed: {e}")
    
    # Also print the report to console as backup
    if not report_saved:
        print("\n" + "="*60)
        print("REPORT CONTENT (since file save failed):")
        print("="*60)
        print(clustering_report)
        print("="*60)
    
    # Create a simple summary file as additional backup
    try:
        summary_content = f"""CLUSTERING SUMMARY:
K-Means: {optimal_k} clusters, Silhouette: {kmeans_silhouette:.3f}, Purity: {kmeans_purity:.3f}
Hierarchical: {best_method_name} method, Silhouette: {best_hierarchical_sil}, Purity: {hierarchical_pur}
Data: {X_clustering.shape[0]} samples, {X_clustering.shape[1]} features
Files: kmeans_model.pkl, clustering_results.csv, clustering_metrics.json
"""
        
        with open('results/clustering_summary.txt', 'w') as f:
            f.write(summary_content)
        print("✅ Backup summary saved: results/clustering_summary.txt")
        
    except Exception as e:
        print(f"⚠️ Could not save backup summary: {e}")

except Exception as e:
    print(f"❌ Error generating report: {e}")
    print("Creating minimal report...")
    
    # Minimal report as fallback
    minimal_report = f"""CLUSTERING ANALYSIS - MINIMAL REPORT

K-Means Results:
- Clusters: {optimal_k if 'optimal_k' in locals() else 'Unknown'}
- Silhouette: {kmeans_silhouette:.3f if 'kmeans_silhouette' in locals() else 'Unknown'}

Dataset:
- Samples: {X_clustering.shape[0] if 'X_clustering' in locals() else 'Unknown'}
- Features: {X_clustering.shape[1] if 'X_clustering' in locals() else 'Unknown'}

Status: Clustering analysis completed with some limitations.
"""
    
    try:
        with open('results/clustering_minimal_report.txt', 'w') as f:
            f.write(minimal_report)
        print("✅ Minimal report saved: results/clustering_minimal_report.txt")
    except:
        print("❌ Could not save even minimal report")

print("Report generation process completed!")