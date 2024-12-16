import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def calculate_attention_distribution(points, scores):
    """
    Calculate the distribution of attention states
    Low attention: 0 < score ≤ 0.050
    High attention: score > 0.050
    """
    # Attention state masks
    low_attention_mask = (0 < scores) & (scores <= 0.050)
    high_attention_mask = scores > 0.050
    
    metrics = {
        'low_attention_points': np.sum(low_attention_mask),
        'high_attention_points': np.sum(high_attention_mask),
        'low_attention_ratio': np.mean(low_attention_mask),
        'high_attention_ratio': np.mean(high_attention_mask),
        'high_to_low_ratio': (np.sum(high_attention_mask) / 
                            np.sum(low_attention_mask) if np.sum(low_attention_mask) > 0 else 0)
    }
    
    return metrics

def analyze_attention_patterns(data_dir):
    """Analyze attention patterns for each demographic"""
    results = []
    
    for demo in ['novice', 'expert']:
        subject_dirs = [d for d in os.listdir(os.path.join(data_dir, demo)) 
                       if d.startswith('subject_')]
        
        for subject_dir in subject_dirs:
            subject_id = subject_dir.split('_')[1]
            
            path = os.path.join(data_dir, demo, subject_dir, 'et')
            
            if not os.path.exists(path):
                continue
            
            for file in sorted(os.listdir(path)):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(path, file))
                    points = df[['x', 'y', 'z']].values
                    scores = df['NormalizedScore'].values
                    
                    metrics = calculate_attention_distribution(points, scores)
                    
                    results.append({
                        'demographic': demo,
                        'subject_id': subject_id,
                        **metrics
                    })
    
    df = pd.DataFrame(results)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'attention_intensity_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    
    # 1. Side-by-side comparison of attention ratios
    plt.figure(figsize=(12, 6))
    df_melted = pd.melt(df, 
                        id_vars=['demographic'], 
                        value_vars=['low_attention_ratio', 'high_attention_ratio'],
                        var_name='attention_type', 
                        value_name='ratio')
    
    sns.boxplot(data=df_melted, x='demographic', y='ratio', 
                hue='attention_type', palette='Set2')
    plt.title('Distribution of High vs Low Attention States')
    plt.ylabel('Ratio of Points')
    plt.savefig(os.path.join(output_dir, 'attention_distribution.png'))
    plt.close()
    
    # 2. High-to-low ratio comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='demographic', y='high_to_low_ratio', palette='Set2')
    plt.title('Ratio of High to Low Attention Points')
    plt.ylabel('High/Low Attention Ratio')
    plt.savefig(os.path.join(output_dir, 'high_to_low_ratio.png'))
    plt.close()
    
    # Statistical analysis
    stats_results = []
    for metric in ['low_attention_ratio', 'high_attention_ratio', 'high_to_low_ratio']:
        novice_data = df[df['demographic'] == 'novice'][metric]
        expert_data = df[df['demographic'] == 'expert'][metric]
        
        statistic, pvalue = stats.mannwhitneyu(
            novice_data, expert_data, alternative='two-sided'
        )
        
        stats_results.append({
            'metric': metric,
            'p_value': pvalue,
            'novice_mean': novice_data.mean(),
            'expert_mean': expert_data.mean(),
            'percent_difference': ((novice_data.mean() - expert_data.mean()) / 
                                 expert_data.mean() * 100)
        })
    
    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(os.path.join(output_dir, 'attention_statistics.csv'), 
                    index=False)
    
    return df, stats_df

if __name__ == "__main__":
    results_df, stats_df = analyze_attention_patterns('data')
    
    print("\nKey Findings:")
    print("-" * 50)
    for _, row in stats_df.iterrows():
        if row['p_value'] < 0.05:
            print(f"\n{row['metric']}:")
            print(f"Novice mean: {row['novice_mean']:.3f}")
            print(f"Expert mean: {row['expert_mean']:.3f}")
            print(f"Difference: {row['percent_difference']:.1f}%")
            print(f"p-value: {row['p_value']:.3f}") 