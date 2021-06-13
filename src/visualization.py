import matplotlib.pyplot as plt

def plot_metrics(df, metrics=None, selected_slice=None, figsize=(14,10)):
    '''
    Plot time series metrics

    Params
    ----
    df: pandas dataframe containing ordered, multivariate timeseries
    metrics: a list containing the metrics to be visualized
    selected_slice: how many subsequent observations to plot
    figsize: size of the plot
    '''

    if metrics is None:
        metrics = df.columns

    fig, ax = plt.subplots(len(metrics), figsize=(14,10), sharex=True)

    for idx, metric in enumerate(metrics):
        if selected_slice is not None:
            df.loc[selected_slice, metric].plot(ax=ax[idx])
        else:
            df.iloc[:,:][metric].plot(ax=ax[idx])
        ax[idx].set_title(metric)

    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    plt.subplots_adjust(hspace=.7)