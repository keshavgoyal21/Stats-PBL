import matplotlib.pyplot as plt
import seaborn as sns

def descriptive_stats(df):
    print("\n MEAN \n", df.mean(numeric_only=True))
    print("\n MEDIAN \n", df.median(numeric_only=True))
    print("\n MODE \n", df.mode(numeric_only=True).iloc[0])

def class_distribution(y):
    sns.countplot(x=y)
    plt.title("Class Distribution")
    plt.savefig("outputs/plots/class_distribution.png")
    plt.show()

def correlation_heatmap(df):
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig("outputs/plots/correlation.png")
    plt.show()

def feature_distribution(df):
    df.hist(figsize=(12,10))
    plt.savefig("outputs/plots/histograms.png")
    plt.show()

def boxplot_outliers(df):
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df.iloc[:, :10])
    plt.xticks(rotation=90)
    plt.savefig("outputs/plots/boxplot.png")
    plt.show()