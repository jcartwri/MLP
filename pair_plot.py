import sys
import matplotlib.pyplot as plot
from describe import DataSet
from describe import Math_calculat
from scatter_plot import Scatter_Plot
from histogram import Histogram
import numpy as np
import argparse

class Pair_Plot:
    """
        - Pair Plot It's matrix plots and scater_plots
        -
        - :param file: train csv file is dataset with feature and target, default is datasets/dataset_train.csv
        - :param max_nb_columns: Max count print numeric columns on the table, default value is 4
        - :param size: size value point wil be on the scatter plot, default is 10
        - :param fig_size: size windows for prints plots, default is 10
        - :param legend: flag legend defined wil be table with description in lower right plot or No, default is 1
        -
        - Example to run:
        >>  pr = Pair_Plot(file=file, max_nb_columns=max_nb_columns, size=size, fig_size=fig_size, legend)
        >>  pr.Plot()
    """
    def __init__(self, file='data.csv', max_nb_columns=3, size=10, fig_size=(8, 8), legend=1):

        self.file_name = file
        self.y_col = 1
        self.max_nb_columns = max_nb_columns
        self.size = size
        self.fig_size = fig_size
        self.legend = legend
        if max_nb_columns < 0 or size < 0:
            print("Error: bad parameter : max_nb_columns or size")
            sys.exit()

    def Plot(self):
        ds = DataSet(self.file_name)
        ds.find_numeric_label()
        if self.max_nb_columns > (len(ds.numeric_columns)):
            self.max_nb_columns = len(ds.numeric_columns)

        color = {
            'M': 'b',
            'B': 'r',
        }

        fig, ax = plot.subplots(self.max_nb_columns, self.max_nb_columns, figsize=self.fig_size)

        fig.tight_layout()
        N = self.max_nb_columns
        for i in range(N):
            col1 = ds.get_float_col(ds.numeric_columns[i])[:self.size]
            for j in range(N):
                col2 = ds.get_float_col(ds.numeric_columns[j])[:self.size]
                feature1 = {}
                feature2 = {}
                for k in set(ds.get_col(self.y_col)):
                    feature1[k] = []
                    feature2[k] = []
                for k in range(1, len(ds.dataset[:self.size])):
                    feature1[ds.dataset[k][self.y_col]].append(col1[k - 1])
                    feature2[ds.dataset[k][self.y_col]].append(col2[k - 1])
                if i == 0:
                    ax[i, j].xaxis.set_label_position('top')
                    ax[i, j].set_xlabel('col = {}'.format(ds.numeric_columns[j]), rotation=0)
#                     ax[i, j].set_xlabel(ds.dataset[0][ds.numeric_columns[j]], rotation=0)
                if j == 0:
                    ax[i, j].set_xlabel('col = {}'.format(ds.numeric_columns[i]), rotation=1)
#                     ax[i, j].set_ylabel(ds.dataset[0][ds.numeric_columns[i]], rotation=0)
                if (i == j):
                    statistic = Math_calculat(col1)
                    bins = np.linspace(statistic.Quartile(0), statistic.Quartile(1))
                    for k in feature1.keys():
                        ax[i, j].hist(feature1[k], bins, facecolor=color[k], alpha=0.5, label=k)

                else:
                    for k in feature1.keys():
                        ax[i, j].scatter(feature1[k], feature2[k], c=color[k], alpha=0.5, label=k)
                ax[i, j].tick_params(labelbottom=False)
                ax[i, j].tick_params(labelleft=False)

        if self.legend:
            plot.legend(loc='lower right')
        plot.savefig('data/pair_plot.png')
        plot.show()

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--file', nargs=1,
                            help="train csv file is dataset with feature and target, default is datasets/dataset_train.csv",
                            default=['data/data.csv'], required=False)
        parser.add_argument('--max_nb_columns', nargs=1,
                            help="Max count print numeric columns on the table, default value is 4",
                            default=[10], type=int, required=False)
        parser.add_argument('--size', nargs=1, help='size value point wil be on the scatter plot, default is 10',
                            default=[1700], type=int, required=False)
        parser.add_argument('--size_window', nargs=1, help='size windows for prints plots, default is 10',
                            default=[17], type=int, required=False)
        parser.add_argument('--legend', nargs=1,
                            help='flag legend defined wil be table with description in lower right plot or No, default is 1',
                            default=[1], type=int, required=False)
        pr = Pair_Plot(file=parser.parse_args().file[0], max_nb_columns=parser.parse_args().max_nb_columns[0], size=parser.parse_args().size[0],
                       fig_size=(parser.parse_args().size_window[0], parser.parse_args().size_window[0]), legend=parser.parse_args().legend[0])
        pr.Plot()
    except Exception:
        print ("Error: Bad parameters")