def adjust_plot_margins():
    """Fix matplotlib bug that cuts off top/bottom of seaborn visualizations"""
    b, t = plt.ylim()
    plt.ylim(b + 0.5, t - 0.5)


def calculate_outliers(df):
    """Calculate number of outliers in each column"""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum().sort_values(ascending=False)
