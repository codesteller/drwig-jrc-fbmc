'''
 # @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
 # @ Author: Pallab Maji
 # @ Create Time: 2025-11-07 13:52:59
 # @ Modified time: 2025-11-07 14:07:01
 # @ Description: Enter description here
 '''
import fbmc_ofdm_validation as fv
import matplotlib.pyplot as plt



def main():
    """Run all validations and generate report"""
    print("\n" + "="*60)
    print("FBMC vs OFDM VALIDATION FRAMEWORK")
    print("For IEEE Paper: FBMC Waveforms for Joint Radar-Communication")
    print("="*60)
    
    # Create all plots
    fig1 = fv.plot_spectral_comparison()
    fig2 = fv.plot_range_doppler_comparison()
    fig3 = fv.plot_ambiguity_functions()
    fig4 = fv.plot_doppler_tolerance()

    # Compute metrics
    metrics = fv.compute_performance_metrics()
    
    # Save all figures
    fig1.savefig('./logs/spectral_comparison.png', dpi=300, bbox_inches='tight')
    fig2.savefig('./logs/range_doppler_comparison.png', dpi=300, bbox_inches='tight')
    fig3.savefig('./logs/ambiguity_comparison.png', dpi=300, bbox_inches='tight')
    fig4.savefig('./logs/doppler_tolerance.png', dpi=300, bbox_inches='tight')
    
    print("\n✓ All validation plots saved successfully!")
    
    # Create metrics table figure
    fig5, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = []
    for i in range(len(metrics['Metric'])):
        table_data.append([metrics['Metric'][i], metrics['OFDM'][i], 
                          metrics['FBMC'][i], metrics['Improvement'][i]])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Performance Metric', 'OFDM', 'FBMC', 'Improvement'],
                    cellLoc='center',
                    loc='center',
                    colColours=['lightgray']*4)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Color improvements
    for i in range(1, len(table_data)+1):
        cell = table[(i, 3)]
        if i in [1, 2, 4]:  # Positive improvements
            cell.set_facecolor('#90EE90')
        elif i == 5:  # Complexity increase
            cell.set_facecolor('#FFB6C1')
    
    plt.title('FBMC vs OFDM: Performance Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    fig5.savefig('./logs/metrics_table.png', dpi=300, bbox_inches='tight')
    
    print("\nValidation complete! Generated files:")
    print("1. spectral_comparison.png")
    print("2. range_doppler_comparison.png") 
    print("3. ambiguity_comparison.png")
    print("4. doppler_tolerance.png")
    print("5. metrics_table.png")
    
    plt.show()
    
    return metrics

if __name__ == "__main__":
    metrics = main()