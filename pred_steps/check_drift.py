import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zenml import step

@step
def plot_drift(num_week):
    # Load saved weekly prediction CSVs
    week1_df = pd.read_csv("./data/week1_emotions.csv")
    week2_df = pd.read_csv("./data/week2_emotions.csv")
    
    # Create emotion labels (you can replace these with actual emotion names)
    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
        'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
        'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    # If you don't have the actual emotion names, use generic labels
    if len(emotion_labels) != len(week1_df):
        emotion_labels = [f'emotion_{i}' for i in range(len(week1_df))]
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'emotion': emotion_labels,
        'week1': week1_df['emotion'].values,
        'week2': week2_df['emotion'].values
    })
    
    # Calculate metrics
    comparison_df['difference'] = comparison_df['week2'] - comparison_df['week1']
    comparison_df['percent_change'] = ((comparison_df['week2'] - comparison_df['week1']) / 
                                     (comparison_df['week1'] + 1e-8)) * 100
    comparison_df['absolute_difference'] = np.abs(comparison_df['difference'])
    
    # Calculate drift metrics
    total_week1 = comparison_df['week1'].sum()
    total_week2 = comparison_df['week2'].sum()
    
    # Handle case where totals are zero
    if total_week1 == 0:
        total_week1 = 1e-8
    if total_week2 == 0:
        total_week2 = 1e-8
    
    comparison_df['week1_proportion'] = comparison_df['week1'] / total_week1
    comparison_df['week2_proportion'] = comparison_df['week2'] / total_week2
    comparison_df['proportion_difference'] = comparison_df['week2_proportion'] - comparison_df['week1_proportion']
    
    # Statistical tests for drift detection (with zero handling)
    try:
        from scipy.stats import chi2_contingency
        
        # Add small constant to avoid zeros in chi-square test
        week1_adj = comparison_df['week1'].values + 1e-6
        week2_adj = comparison_df['week2'].values + 1e-6
        
        contingency_table = np.array([week1_adj, week2_adj])
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    except Exception as e:
        print(f"Chi-square test failed: {e}")
        chi2_stat, p_value = 0, 1.0
    
    # Wasserstein distance (Earth Mover's Distance) with zero handling
    try:
        from scipy.stats import wasserstein_distance
        wasserstein_dist = wasserstein_distance(comparison_df['week1'], comparison_df['week2'])
    except Exception as e:
        print(f"Wasserstein distance failed: {e}")
        # Simple alternative distance measure
        wasserstein_dist = np.sum(np.abs(comparison_df['difference'])) / len(comparison_df)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Side-by-side comparison
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(emotion_labels))
    width = 0.35
    bars1 = ax1.bar(x - width/2, comparison_df['week1'], width, label='Week 1', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, comparison_df['week2'], width, label='Week 2', alpha=0.8, color='lightcoral')
    ax1.set_xlabel('Emotions')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Emotion Frequencies: Week 1 vs Week 2', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(emotion_labels, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars (only for non-zero values to avoid clutter)
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=6)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=6)
    
    # 2. Drift statistics box
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = f"""
    DRIFT ANALYSIS SUMMARY
    
    📊 Total Emotions: {len(emotion_labels)}
    
    📈 Increased: {len(comparison_df[comparison_df['difference'] > 0])}
    📉 Decreased: {len(comparison_df[comparison_df['difference'] < 0])}
    ➡️  Unchanged: {len(comparison_df[comparison_df['difference'] == 0])}
    
    🔍 Statistical Tests:
    Chi-square: {chi2_stat:.2f}
    P-value: {p_value:.4f}
    Drift Detected: {'Yes' if p_value < 0.05 else 'No'}
    
    📏 Distance Metric: {wasserstein_dist:.2f}
    
    📊 Total Week 1: {int(total_week1)}
    📊 Total Week 2: {int(total_week2)}
    📊 Total Change: {int(total_week2 - total_week1):+d}
    """
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 3. Difference heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in comparison_df['difference']]
    bars = ax3.barh(emotion_labels, comparison_df['difference'], color=colors, alpha=0.7)
    ax3.set_xlabel('Frequency Difference (Week2 - Week1)')
    ax3.set_title('Emotion Changes', fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 4. Proportion comparison (with zero handling)
    ax4 = fig.add_subplot(gs[1, 1])
    # Only plot points where at least one week has non-zero values
    mask = (comparison_df['week1'] > 0) | (comparison_df['week2'] > 0)
    if mask.any():
        scatter = ax4.scatter(comparison_df.loc[mask, 'week1_proportion'], 
                             comparison_df.loc[mask, 'week2_proportion'], 
                             s=100, alpha=0.7, c=comparison_df.loc[mask, 'absolute_difference'], 
                             cmap='Reds')
        max_prop = max(comparison_df['week1_proportion'].max(), comparison_df['week2_proportion'].max())
        ax4.plot([0, max_prop], [0, max_prop], 'r--', alpha=0.5, label='No change line')
        ax4.set_xlabel('Week 1 Proportion')
        ax4.set_ylabel('Week 2 Proportion')
        ax4.set_title('Proportional Changes', fontweight='bold')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Proportional Changes - No Data', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Top changers (only non-zero changes)
    ax5 = fig.add_subplot(gs[1, 2])
    non_zero_changes = comparison_df[comparison_df['absolute_difference'] > 0]
    if len(non_zero_changes) > 0:
        top_changes = non_zero_changes.nlargest(min(10, len(non_zero_changes)), 'absolute_difference')
        colors_top = ['red' if x < 0 else 'green' for x in top_changes['difference']]
        ax5.barh(range(len(top_changes)), top_changes['difference'], color=colors_top, alpha=0.7)
        ax5.set_yticks(range(len(top_changes)))
        ax5.set_yticklabels(top_changes['emotion'], fontsize=8)
        ax5.set_xlabel('Change')
        ax5.set_title(f'Top {len(top_changes)} Biggest Changes', fontweight='bold')
        ax5.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    else:
        ax5.text(0.5, 0.5, 'No changes detected', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('No Changes Detected', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Distribution comparison (with better binning for sparse data)
    ax6 = fig.add_subplot(gs[2, :])
    # Use fewer bins for sparse data
    max_val = max(comparison_df['week1'].max(), comparison_df['week2'].max())
    bins = min(10, max_val + 1) if max_val > 0 else 2
    
    ax6.hist(comparison_df['week1'], bins=bins, alpha=0.5, label='Week 1', density=True, color='skyblue')
    ax6.hist(comparison_df['week2'], bins=bins, alpha=0.5, label='Week 2', density=True, color='lightcoral')
    ax6.set_xlabel('Frequency')
    ax6.set_ylabel('Density')
    ax6.set_title('Frequency Distribution Comparison', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Emotion Drift Analysis - Week Comparison', fontsize=16, fontweight='bold')
    plt.savefig('./data/emotion_drift_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed analysis
    detailed_analysis = comparison_df.sort_values('absolute_difference', ascending=False)
    detailed_analysis.to_csv('./data/emotion_drift_detailed.csv', index=False)
    
    # Create summary report
    summary_report = f"""
    EMOTION DRIFT ANALYSIS REPORT
    ============================
    
    Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    OVERVIEW:
    - Total emotions analyzed: {len(emotion_labels)}
    - Emotions with increased frequency: {len(comparison_df[comparison_df['difference'] > 0])}
    - Emotions with decreased frequency: {len(comparison_df[comparison_df['difference'] < 0])}
    - Emotions with no change: {len(comparison_df[comparison_df['difference'] == 0])}
    
    STATISTICAL ANALYSIS:
    - Chi-square statistic: {chi2_stat:.4f}
    - P-value: {p_value:.6f}
    - Drift detected: {'Yes (p < 0.05)' if p_value < 0.05 else 'No (p >= 0.05)'}
    - Distance metric: {wasserstein_dist:.4f}
    
    TOP 5 EMOTIONS WITH BIGGEST INCREASES:
    {detailed_analysis.head(5)[['emotion', 'week1', 'week2', 'difference', 'percent_change']].to_string(index=False)}
    
    TOP 5 EMOTIONS WITH BIGGEST DECREASES:
    {detailed_analysis.tail(5)[['emotion', 'week1', 'week2', 'difference', 'percent_change']].to_string(index=False)}
    
    FILES GENERATED:
    - emotion_drift_analysis.png (visualization)
    - emotion_drift_detailed.csv (detailed data)
    - emotion_drift_report.txt (this report)
    """
    
    with open('./data/emotion_drift_report.txt', 'w') as f:
        f.write(summary_report)
    
    print("✅ Analysis complete!")
    print("📊 Generated files:")
    print("   - ./data/emotion_drift_analysis.png")
    print("   - ./data/emotion_drift_detailed.csv") 
    print("   - ./data/emotion_drift_report.txt")
    print(f"\n🔍 Drift detected: {'Yes' if p_value < 0.05 else 'No'}")
    print(f"📏 Distance metric: {wasserstein_dist:.4f}")
    
    return comparison_df