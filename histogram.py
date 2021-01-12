import csv
import math
import argparse
import matplotlib.pyplot as plot
import numpy as np
import sys
from describe import Math_calculat
from describe import DataSet

class Histogram:
    """
            - class Histogram allow to built grafic Histogram on the dataset, Frequency data from class house.
            - It's answer on the question: Which Hogwarts course has a homogeneous score distribution between all four houses?

            - Parameters:
            -   file : str - It's path to csv file, dataset. Default value is 'datasets/dataset_train.csv'
            -   granularity : int -  It's granularity value feature, allow value from 0 to inf. Default value is 100
            -   col_nb  - It's name or index column with value course. Default value is 'Arithmancy'

            - Example to run:
                >> from histogram import Histogram
                >> import matplotlib.pyplot as plot
                >> hs = Histogram(file='dataset.csv', granularity = 100)
                >> hs.Plot(col_nb=6)
    """
    def __init__(self, file='data.csv', size=100, legend=1):
        self.file_name = file
        self.y_col = 1
        self.size = size
        self.col_nb = None
        self.legend = legend

        if size < 0:
            print("Error with granularity")
            sys.exit()

    def Plot(self, col_nb):
        ds = DataSet(self.file_name)
        if (type(col_nb) is str):
            if col_nb in ds.dataset[0]:
                col_nb = ds.dataset[0].index(col_nb)
            else:
                print('Error with name column')
                return

        if not ds.isNumeric_columns(col_nb):
            print("Input column must is numerics")
            return

        col = ds.get_float_col(col_nb)
        statistic = Math_calculat(col)
        bins = np.linspace(statistic.Quartile(0), statistic.Quartile(1), self.size)
        color = {
            'M': 'b',
            'B': 'r',
        }

        feature = {}
        for i in set(ds.get_col(self.y_col)): feature[i] = []
        for i in range(1, len(ds.dataset)):
            feature[ds.dataset[i][self.y_col]].append(col[i - 1])

        for i in feature.keys():
            plot.hist(feature[i], bins, facecolor=color[i], alpha=0.5, label=i)
        if self.legend:
            plot.legend(loc='upper right')
        plot.ylabel('Frequency')
        plot.xlabel('Value')
        plot.title('Histogram, column is {}'.format(col_nb))
        plot.savefig('data/histogram.png')
        plot.show()

if __name__ == '__main__':
    # try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--file', nargs=1, help="train csv file is dataset with feature and target, default is data.csv",
                            default=['data/data.csv'], required=False)
        parser.add_argument('--course', nargs=1, help="Which will be column, default is 2",
                            default=[2], type=int, required=False)
        parser.add_argument('--size', nargs=1, help='size value on the histogram, default is 100',
                            default=[100], type=int, required=False)
        parser.add_argument('--legend', nargs=1, help='flag legend defined wil be table with description in lower right plot or No, default is 1',
                    default=[1], type=int, required=False)

        hs = Histogram(file=parser.parse_args().file[0], size=parser.parse_args().size[0], legend=parser.parse_args().legend[0])
        hs.Plot(int(parser.parse_args().course[0]))

