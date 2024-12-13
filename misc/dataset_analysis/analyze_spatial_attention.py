import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

def calculate_attention_metrics(points, scores, config_type):
    """
    Calculate metrics using the established class thresholds
    
    ET Classes:
    0: x = 0 (No Attention)
    1: 0 < x ≤ 0.025 (Very Low)
    2: 0.025 < x ≤ 0.050 (Low)
    3: 0.050 < x ≤ 0.1 (Medium)
    4: x > 0.1 (High)
    
    EEG Classes:
    0: x ≤ -0.5 (Very negative)
    1: -0.5 < x ≤ 0 (Slightly negative)
    2: x = 0 (No attention)
    3: 0 < x ≤ 0.5 (Slightly positive)
    4: x > 0.5 (Very positive)
    """
    if config_type == 'et':
        # Create masks for each attention class
        no_attention_mask = scores == 0
        very_low_mask = (0 < scores) & (scores <= 0.025)
        low_mask = (0.025 < scores) & (scores <= 0.050)
        medium_mask = (0.050 < scores) & (scores <= 0.1)
        high_mask = scores > 0.1
        
        # Points with meaningful attention (class 2 and above - Low and higher)
        attention_mask = scores > 0.025
        attention_points = points[attention_mask]
        attention_scores = scores[attention_mask]
        
        metrics = {
            'no_attention_ratio': np.mean(no_attention_mask),
            'very_low_ratio': np.mean(very_low_mask),
            'low_ratio': np.mean(low_mask),
            'medium_ratio': np.mean(medium_mask),
            'high_ratio': np.mean(high_mask),
            'meaningful_attention_ratio': np.mean(attention_mask)  # Low and above
        }
        
    else:  # eeg
        # Create masks for each preference class
        very_negative_mask = scores <= -0.5
        slightly_negative_mask = (-0.5 < scores) & (scores <= 0)
        no_attention_mask = scores == 0
        slightly_positive_mask = (0 < scores) & (scores <= 0.5)
        very_positive_mask = scores > 0.5
        
        # Any non-zero attention
        attention_mask = scores != 0
        attention_points = points[attention_mask]
        attention_scores = scores[attention_mask]
        
        metrics = {
            'very_negative_ratio': np.mean(very_negative_mask),
            'slightly_negative_ratio': np.mean(slightly_negative_mask),
            'no_attention_ratio': np.mean(no_attention_mask),
            'slightly_positive_ratio': np.mean(slightly_positive_mask),
            'very_positive_ratio': np.mean(very_positive_mask),
            'total_attention_ratio': np.mean(attention_mask)
        }
    
    if len(attention_points) == 0:
        metrics.update({
            'attention_density': 0,
            'attention_clusters': 0,
            'attention_dispersion': 0
        })
        return metrics
    
    # Calculate spatial metrics for points with attention
    try:
        hull = ConvexHull(points)
        attention_hull = ConvexHull(attention_points)
        attention_density = attention_hull.volume / hull.volume
    except:
        attention_density = 0
    
    # Find attention clusters
    model_size = np.ptp(points, axis=0).max()
    eps = model_size * 0.1
    
    if config_type == 'et':
        # For ET, cluster points with meaningful attention (Low and above)
        clustering = DBSCAN(eps=eps, min_samples=3).fit(attention_points)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    else:  # eeg
        # For EEG, separate clusters for positive and negative attention
        positive_points = points[scores > 0]
        negative_points = points[scores < 0]
        
        pos_clusters = DBSCAN(eps=eps, min_samples=3).fit(positive_points) if len(positive_points) > 0 else None
        neg_clusters = DBSCAN(eps=eps, min_samples=3).fit(negative_points) if len(negative_points) > 0 else None
        
        pos_n_clusters = len(set(pos_clusters.labels_)) - (1 if -1 in pos_clusters.labels_ else 0) if pos_clusters is not None else 0
        neg_n_clusters = len(set(neg_clusters.labels_)) - (1 if -1 in neg_clusters.labels_ else 0) if neg_clusters is not None else 0
        n_clusters = pos_n_clusters + neg_n_clusters
    
    # Calculate dispersion from centroid
    centroid = np.mean(attention_points, axis=0)
    distances = np.linalg.norm(attention_points - centroid, axis=1)
    attention_dispersion = np.mean(distances)
    
    metrics.update({
        'attention_density': attention_density,
        'attention_clusters': n_clusters,
        'attention_dispersion': attention_dispersion
    })
    
    return metrics

def analyze_spatial_attention(data_dir):
    """Analyze spatial distribution of attention for each demographic"""
    results = []
    
    for demo in ['novice', 'expert']:
        subject_dirs = [d for d in os.listdir(os.path.join(data_dir, demo)) 
                       if d.startswith('subject_')]
        
        for subject_dir in subject_dirs:
            subject_id = subject_dir.split('_')[1]
            
            for config in ['et', 'eeg']:
                path = os.path.join(data_dir, demo, subject_dir, config)
                
                if not os.path.exists(path):
                    continue
                
                for file in sorted(os.listdir(path)):
                    if file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(path, file))
                        points = df[['x', 'y', 'z']].values
                        
                        score_col = 'NormalizedEEGScore' if config == 'eeg' else 'NormalizedScore'
                        scores = df[score_col].values
                        
                        metrics = calculate_attention_metrics(points, scores, config)
                        
                        results.append({
                            'demographic': demo,
                            'subject_id': subject_id,
                            'config_type': config,
                            'model': file.replace('.csv', ''),
                            'form_type': 'curved' if 'curved' in file else 'rect',
                            **metrics
                        })
    
    df = pd.DataFrame(results)
    
    # Create output directory for visualizations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'spatial_attention_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations for ET and EEG
    for config in ['et', 'eeg']:
        config_df = df[df['config_type'] == config]
        
        # Plot class distributions
        if config == 'et':
            class_metrics = ['no_attention_ratio', 'very_low_ratio', 'low_ratio', 'medium_ratio', 'high_ratio']
            class_labels = ['No Attention', 'Very Low', 'Low', 'Medium', 'High']
        else:  # eeg
            class_metrics = ['very_negative_ratio', 'slightly_negative_ratio', 'no_attention_ratio', 
                           'slightly_positive_ratio', 'very_positive_ratio']
            class_labels = ['Very Negative', 'Slightly Negative', 'No Attention', 
                          'Slightly Positive', 'Very Positive']
        
        # Create stacked bar plot for class distributions
        plt.figure(figsize=(12, 6))
        data_to_plot = config_df.groupby('demographic')[class_metrics].mean()
        data_to_plot.plot(kind='bar', stacked=True)
        plt.title(f'Distribution of {config.upper()} Classes by Demographic')
        plt.xlabel('Demographic')
        plt.ylabel('Ratio')
        plt.legend(class_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{config}_class_distribution.png'))
        plt.close()
        
        # Plot spatial metrics
        spatial_metrics = ['attention_density', 'attention_clusters', 'attention_dispersion']
        for metric in spatial_metrics:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=config_df, x='form_type', y=metric, 
                       hue='demographic', palette='Set2')
            plt.title(f'{metric.replace("_", " ").title()} by Demographic ({config.upper()})')
            plt.savefig(os.path.join(output_dir, f'{metric}_{config}_comparison.png'))
            plt.close()
    
    # Statistical analysis
    stats_results = []
    
    for config in ['et', 'eeg']:
        if config == 'et':
            metrics_to_test = ['attention_density', 'attention_clusters', 'attention_dispersion',
                             'no_attention_ratio', 'very_low_ratio', 'low_ratio', 
                             'medium_ratio', 'high_ratio']
        else:  # eeg
            metrics_to_test = ['attention_density', 'attention_clusters', 'attention_dispersion',
                             'very_negative_ratio', 'slightly_negative_ratio', 'no_attention_ratio',
                             'slightly_positive_ratio', 'very_positive_ratio']
        
        for metric in metrics_to_test:
            for form in ['curved', 'rect']:
                novice_data = df[(df['demographic'] == 'novice') & 
                               (df['config_type'] == config) &
                               (df['form_type'] == form)][metric]
                expert_data = df[(df['demographic'] == 'expert') &
                               (df['config_type'] == config) &
                               (df['form_type'] == form)][metric]
                
                if len(novice_data) == 0 or len(expert_data) == 0:
                    continue
                
                statistic, pvalue = stats.mannwhitneyu(
                    novice_data, expert_data, alternative='two-sided'
                )
                
                stats_results.append({
                    'config_type': config,
                    'form_type': form,
                    'metric': metric,
                    'p_value': pvalue,
                    'novice_mean': novice_data.mean(),
                    'expert_mean': expert_data.mean(),
                    'percent_difference': ((novice_data.mean() - expert_data.mean()) / 
                                         expert_data.mean() * 100)
                })
    
    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(os.path.join(output_dir, 'spatial_attention_statistics.csv'), 
                    index=False)
    
    return df, stats_df

if __name__ == "__main__":
    results_df, stats_df = analyze_spatial_attention('data')
    
    print("\nKey Findings:")
    print("-" * 50)
    for _, row in stats_df.iterrows():
        if row['p_value'] < 0.05:
            print(f"\n{row['config_type'].upper()} - {row['metric']} ({row['form_type']}):")
            print(f"Novice mean: {row['novice_mean']:.3f}")
            print(f"Expert mean: {row['expert_mean']:.3f}")
            print(f"Difference: {row['percent_difference']:.1f}%")
            print(f"p-value: {row['p_value']:.3f}")
