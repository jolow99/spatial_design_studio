import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def get_form_type(filename):
    """Extract form type from filename"""
    return 'curvilinear' if filename.startswith('curved') else 'rectilinear'

def get_attention_class(scores, config_type='et'):
    """Classify attention/preference scores into categories"""
    classes = np.zeros_like(scores, dtype=int)
    
    if config_type == 'eeg':
        classes[scores <= -0.5] = 0  # Very negative
        classes[(-0.5 < scores) & (scores <= 0)] = 1  # Slightly negative
        classes[scores == 0] = 2  # No attention
        classes[(0 < scores) & (scores <= 0.5)] = 3  # Slightly positive
        classes[scores > 0.5] = 4  # Very positive
    else:  # et
        classes[scores == 0] = 0  # No Attention
        classes[(0 < scores) & (scores <= 0.025)] = 1  # Very Low
        classes[(0.025 < scores) & (scores <= 0.050)] = 2  # Low
        classes[(0.050 < scores) & (scores <= 0.1)] = 3  # Medium
        classes[scores > 0.1] = 4  # High
    
    return classes

def analyze_form_preferences(data_dir):
    """Analyze attention and preference patterns for different form types"""
    results = []
    
    for demo in ['novice', 'expert']:
        subject_dirs = [d for d in os.listdir(os.path.join(data_dir, demo)) 
                       if d.startswith('subject_')]
        
        for subject_dir in subject_dirs:
            subject_id = subject_dir.split('_')[1]
            
            # Analyze both ET and EEG data
            for data_type in ['et', 'eeg']:
                path = os.path.join(data_dir, demo, subject_dir, data_type)
                
                if not os.path.exists(path):
                    continue
                
                for file in sorted(os.listdir(path)):
                    if file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(path, file))
                        form_type = get_form_type(file)
                        
                        # Calculate metrics
                        if data_type == 'et':
                            score_col = 'NormalizedScore'
                            scores = df[score_col].values
                            classes = get_attention_class(scores, 'et')
                            
                            results.append({
                                'demographic': demo,
                                'subject_id': subject_id,
                                'form_type': form_type,
                                'data_type': data_type,
                                'mean_attention': scores.mean(),
                                'high_attention_ratio': np.mean(scores > 0.1),  # Class 4
                                'medium_attention_ratio': np.mean((0.050 < scores) & (scores <= 0.1)),  # Class 3
                                'low_attention_ratio': np.mean((0 < scores) & (scores <= 0.050)),  # Classes 1-2
                                'class_distribution': np.bincount(classes, minlength=5) / len(classes),
                                'filename': file
                            })
                        
                        else:  # EEG
                            score_col = 'NormalizedEEGScore'
                            scores = df[score_col].values
                            classes = get_attention_class(scores, 'eeg')
                            
                            results.append({
                                'demographic': demo,
                                'subject_id': subject_id,
                                'form_type': form_type,
                                'data_type': data_type,
                                'mean_preference': scores.mean(),
                                'very_positive_ratio': np.mean(scores > 0.5),  # Class 4
                                'slightly_positive_ratio': np.mean((0 < scores) & (scores <= 0.5)),  # Class 3
                                'negative_ratio': np.mean(scores < 0),  # Classes 0-1
                                'class_distribution': np.bincount(classes, minlength=5) / len(classes),
                                'filename': file
                            })
    
    df = pd.DataFrame(results)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'form_type_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    
    # 1. ET: Mean attention by form type and demographic
    plt.figure(figsize=(10, 6))
    et_data = df[df['data_type'] == 'et']
    sns.boxplot(data=et_data, x='demographic', y='mean_attention', 
                hue='form_type', palette='Set2')
    plt.title('Mean Attention by Form Type and Demographic (ET)')
    plt.savefig(os.path.join(output_dir, 'et_attention_by_form.png'))
    plt.close()
    
    # 2. EEG: Mean preference by form type and demographic
    plt.figure(figsize=(10, 6))
    eeg_data = df[df['data_type'] == 'eeg']
    sns.boxplot(data=eeg_data, x='demographic', y='mean_preference', 
                hue='form_type', palette='Set2')
    plt.title('Mean Preference by Form Type and Demographic (EEG)')
    plt.savefig(os.path.join(output_dir, 'eeg_preference_by_form.png'))
    plt.close()
    
    # Add class distribution visualizations
    plt.figure(figsize=(12, 6))
    et_data = df[df['data_type'] == 'et']
    class_cols = ['class_distribution']
    class_names = ['No Attention', 'Very Low', 'Low', 'Medium', 'High']
    
    for i, demo in enumerate(['novice', 'expert']):
        plt.subplot(1, 2, i+1)
        demo_data = et_data[et_data['demographic'] == demo]
        
        curv_dist = np.mean([d for d, ft in zip(demo_data['class_distribution'], 
                           demo_data['form_type']) if ft == 'curvilinear'], axis=0)
        rect_dist = np.mean([d for d, ft in zip(demo_data['class_distribution'], 
                           demo_data['form_type']) if ft == 'rectilinear'], axis=0)
        
        x = np.arange(len(class_names))
        width = 0.35
        
        plt.bar(x - width/2, curv_dist, width, label='Curvilinear')
        plt.bar(x + width/2, rect_dist, width, label='Rectilinear')
        plt.title(f'{demo.capitalize()} ET Class Distribution')
        plt.xlabel('Attention Class')
        plt.ylabel('Proportion')
        plt.xticks(x, class_names, rotation=45)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'et_class_distribution.png'))
    plt.close()

    # Similar visualization for EEG data
    plt.figure(figsize=(12, 6))
    eeg_data = df[df['data_type'] == 'eeg']
    class_names = ['Very Negative', 'Slightly Negative', 'No Attention', 
                  'Slightly Positive', 'Very Positive']
    
    # ... (similar code for EEG class distribution) ...

    # Statistical analysis
    stats_results = []
    
    # Analyze ET data
    for demo in ['novice', 'expert']:
        demo_et = et_data[et_data['demographic'] == demo]
        curv_attention = demo_et[demo_et['form_type'] == 'curvilinear']['mean_attention']
        rect_attention = demo_et[demo_et['form_type'] == 'rectilinear']['mean_attention']
        
        statistic, pvalue = stats.mannwhitneyu(
            curv_attention, rect_attention, alternative='two-sided'
        )
        
        stats_results.append({
            'comparison': f'ET_{demo}',
            'metric': 'mean_attention',
            'p_value': pvalue,
            'curvilinear_mean': curv_attention.mean(),
            'rectilinear_mean': rect_attention.mean(),
            'percent_difference': ((curv_attention.mean() - rect_attention.mean()) / 
                                 rect_attention.mean() * 100)
        })
    
    # Analyze EEG data
    for demo in ['novice', 'expert']:
        demo_eeg = eeg_data[eeg_data['demographic'] == demo]
        curv_pref = demo_eeg[demo_eeg['form_type'] == 'curvilinear']['mean_preference']
        rect_pref = demo_eeg[demo_eeg['form_type'] == 'rectilinear']['mean_preference']
        
        statistic, pvalue = stats.mannwhitneyu(
            curv_pref, rect_pref, alternative='two-sided'
        )
        
        stats_results.append({
            'comparison': f'EEG_{demo}',
            'metric': 'mean_preference',
            'p_value': pvalue,
            'curvilinear_mean': curv_pref.mean(),
            'rectilinear_mean': rect_pref.mean(),
            'percent_difference': ((curv_pref.mean() - rect_pref.mean()) / 
                                 rect_pref.mean() * 100)
        })
    
    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(os.path.join(output_dir, 'form_type_statistics.csv'), 
                    index=False)
    
    return df, stats_df

if __name__ == "__main__":
    results_df, stats_df = analyze_form_preferences('data')
    
    print("\nKey Findings:")
    print("-" * 50)
    for _, row in stats_df.iterrows():
        print(f"\n{row['comparison']}:")
        print(f"Curvilinear mean: {row['curvilinear_mean']:.3f}")
        print(f"Rectilinear mean: {row['rectilinear_mean']:.3f}")
        print(f"Difference: {row['percent_difference']:.1f}%")
        print(f"p-value: {row['p_value']:.3f}")
