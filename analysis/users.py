"""
This module contains quick functions to plot data about the users
"""
import matplotlib.pyplot as plt
import numpy as np

from recommendations_with_IBM.read_data import reader


# load the user activities "catalogue"
u = reader.UserList()

# common kwargs for plots
hist_args = {'bins': np.arange(0, 100, 5), 'rwidth': .7}


def _plot_user_histogram(new_figure=0, activities=True, **kwargs):
    """ Plots: histogram of the number of activities / readings by each users """
    if activities:
        temp = u.df.groupby('user_id').title.count()
        lab = 'activities'
    else:
        temp = u.user_matrix.sum(axis=1)
        lab = 'read articles'

    if new_figure:
        plt.figure(**kwargs)
    plt.hist(temp, **hist_args)
    plt.title(f"Histogram of the number of {lab} by user")
    plt.xlabel(f'nr of {lab}')
    plt.ylabel('count')


def plot_user_activities_histogram(new_figure=0, **kwargs):
    """ Plots: histogram of the number of activities by each users (reading * times an article accounted as *)
    See Also:
        _plot_user_histogram
    """
    _plot_user_histogram(new_figure, activities=True, **kwargs)


def plot_user_unique_readings_histogram(new_figure=0, **kwargs):
    """ Plots: histogram of the number of articles read by users (reading * times an article accounted as 1)
    See Also:
        _plot_user_histogram
    """
    _plot_user_histogram(new_figure, activities=False, **kwargs)


def plot_article_readings_histogram(new_figure=0, **kwargs):
    """ Plots: histogram of the number of users reading each article """
    if new_figure:
        plt.figure(**kwargs)
    plt.hist(u.df.groupby('article_id').email.count(), **hist_args)


def _plot_user_stats(new_figure=0, n_max=50, kind='bar', activities=True, **kwargs):
    """ Plots: plot of the user id vs the number of activities or reading
    Args:
        new_figure:
        n_max: max number of user to visualize
        kind: if 'bar' will plot an horizontal bar plot, else, a regular plot
        **kwargs:
    """
    if activities:
        ylab = 'activities'
        temp = u.df.groupby('user_id').title.count().sort_values(ascending=False)
    else:
        ylab = 'read articles'
        temp = u.user_matrix.sum(axis=1).sort_values(ascending=False)
    if new_figure:
        plt.figure(**kwargs)

    x = [str(xx) for xx in temp.index.tolist()[: n_max]]
    y = temp.values.tolist()[: n_max]
    if kind == 'bar':
        plt.barh(y=x[: : -1], width=y[: : -1])
        plt.yticks(rotation='-30')
        plt.ylabel('User id')
        plt.xlabel(f'Number of {ylab}')
    else:
        plt.plot(x, y, '-o')
        plt.xticks(rotation='-30')
        plt.xlabel('User id')
        plt.ylabel(f'Number of {ylab}')
        plt.grid(True)


def plot_user_activities(new_figure=0, n_max=50, kind='bar', **kwargs):
    """ Plots: plot of the user id vs the number of activities (reading * times an article accounted as *)
    See Also:
        _plot_user_stats
    """
    _plot_user_stats(new_figure, n_max, kind=kind, activities=True, **kwargs)


def plot_user_articles(new_figure=0, n_max=50, kind='bar', **kwargs):
    """ Plots: plot of the user id vs the number of articles (reading * times an article accounted as 1)
    See Also:
        _plot_user_stats
    """
    _plot_user_stats(new_figure, n_max, kind=kind, activities=False, **kwargs)
