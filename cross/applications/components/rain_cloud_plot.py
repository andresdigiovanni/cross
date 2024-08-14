import matplotlib.pyplot as plt
import seaborn as sns

from cross.applications.styles import plot_remove_borders


def rain_cloud_plot(df, column, figsize=(4, 2), color="#FF4C4B"):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    boxplot_props = {
        "boxprops": {"facecolor": "none"},
    }

    sns.histplot(df[column], kde=True, ax=ax1, color=color)
    # sns.violinplot(x=df[column], ax=ax1, inner=None, cut=0, split=True, color=color)
    sns.stripplot(x=df[column], ax=ax2, size=2, jitter=1, color=color)
    sns.boxplot(x=df[column], ax=ax2, color=color, **boxplot_props)

    for ax in [ax1, ax2]:
        plot_remove_borders(ax)

    ax1.xaxis.label.set_visible(False)

    ax1.yaxis.set_ticks([])
    ax2.yaxis.set_ticks([])

    plt.subplots_adjust(hspace=0)
    plt.tight_layout()

    return fig
