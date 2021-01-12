import math
import csv
import argparse


class Math_calculat:
    """
        - It's class for calculate different operations over list

        - Parameters:
        -    mas: list - It's list with them wil be work.
        -    value: int - It's perceptil, allow value from 0 to 1. Default value is 0.25

        - Example to run:
            >> calc = Math_calculat([1,2,3,4,5,6])
            >> print (calc.Mean()) // mean array
            >> print (calc.Sum()) // sum array
            >> print (calc.Std()) // standard deviation (or std) array
            >> print (calc.Quartile(value))
            >> print (calc.Count()) // count not null element array
            >> print (calc.Sort()) // sort value array from min to max and dell null value
    """
    def __init__(self, mas: list):
        self.mas = mas
        self.sorted = False

    def Mean(self):
        return sum(self.mas) / self.Count()

    def Sum(self):
        return sum(self.mas)

    def Std(self):
        massiv = [i for i in self.mas if i]
        mean = self.Mean()
        stat = Math_calculat([(i - mean) ** 2 for i in massiv])
        s = stat.Sum()
        n = stat.Count()
        return (math.sqrt(s / n))

    def Count(self, massiv=None) -> int:
        return len([x for x in self.mas if x])

    def Sort(self):
        if (not self.sorted):
            self.mas = [i for i in self.mas if i]
            self.mas.sort()
            self.sorted = True

    def Quartile(self, perceptil=0.25):
        self.Sort()
        n = len(self.mas) - 1
        if perceptil * n % 1 == 0:
            return self.mas[math.floor(perceptil * n)]
        else:
            return (self.mas[math.floor(perceptil * n)] + self.mas[math.floor(perceptil * n) + 1]) / 2

class DataSet:
    """
        - It's class allow read and store dataset from csv file

        - Parameters:
        -    filename : str - It's path to csv file with dataset
        -    full_output : bool It's flag for full output numeric columns or short output (where null value less 10 %
                                from all value this column. Default value is False :short output
        -    delim : str - It's char for delimiter csv file

        - Example to run:
            >> from describe import Dataset
            >> from describe import Math_calculat
            >> df = Dataset(filename = 'dataset.csv', delim = ',')
            >> ds.find_numeric_label() // find numeric columns from dataset
            >> ds.output_describe() // output describe about numeric columns
    """
    def __init__(self, filename, full_output=False, delim=','):
        self.dataset = []
        self.full_output = full_output
        self.describe = None
        with open(filename, 'r') as f:
            spamreader = csv.reader(f, delimiter=delim)
            for row in spamreader:
                self.dataset.append(row)

    def isNumeric_columns(self, col):
        empty = 0
        num = 0

        for data in range(1, len(self.dataset)):
            if self.dataset[data][col] == '':
                empty += 1
            else:
                num += self.dataset[data][col].replace('-', '', 1).replace('.', '', 1).isdigit()
        
        if len(self.dataset) == empty:
            return False
        if self.full_output:
            if (empty + num == (len(self.dataset) - 1)):
                return True
        else:
            if (num / (len(self.dataset) - empty) > 0.9):
                return True
        return False

    def find_numeric_label(self):
        self.numeric_columns = []
        for i in range(1, len(self.dataset[0])):
            if self.isNumeric_columns(i):
                self.numeric_columns.append(i)

    def get_float_col(self, i):
        col = []
        for row in range(1, len(self.dataset)):
            if self.dataset[row][i] == '':
                col.append(0.0)
            else:
                col.append(float(self.dataset[row][i]))
        return col

    def get_col(self, i):
        col = []
        for row in range(1, len(self.dataset)):
            if self.dataset[row][i] == '':
                col.append('')
            else:
                col.append(self.dataset[row][i])
        return col

    def fill_describe(self):
        self.describe = {}
        if not self.numeric_columns:
            return
        for i in self.numeric_columns:
            elem_desc = {}
            col = self.get_float_col(i)
            calculate = Math_calculat(col)
            elem_desc['Count'] = calculate.Count()
            elem_desc['Mean'] = calculate.Mean()
            elem_desc['Std'] = calculate.Std()
            elem_desc['Min'] = calculate.Quartile(0)
            elem_desc['25%'] = calculate.Quartile(0.25)
            elem_desc['50%'] = calculate.Quartile(0.5)
            elem_desc['75%'] = calculate.Quartile(0.75)
            elem_desc['Max'] = calculate.Quartile(1)
            self.describe[i] = elem_desc

    def output_describe(self):
        self.fill_describe()
        len_win = max([len(self.dataset[0][i]) for i in self.numeric_columns])
        if len_win <= 22:
            col_bloc = 4
        elif len_win <= 28:
            col_bloc = 3
        elif len_win <= 38:
            col_bloc = 2
        else:
            col_bloc = 1
        if len_win < 20: len_win = 20
        template = '{:<%d}|' % len_win
        col_out = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

        bloc = [self.numeric_columns[d: d + col_bloc] for d in range(0, len(self.numeric_columns), col_bloc)]
        for k in bloc:
            print('\n\n')
            print(('|' + '-' * len_win) * (col_bloc + 1) + '|')
            print('|' + template.format('Feature'), end='')
            for l in k:
                print(template.format(self.dataset[0][l]), end='')
            print('\n' + ('|' + '-' * len_win) * (col_bloc + 1) + '|')
            for i in col_out:
                print('|' + template.format(i), end="")
                for l in k:
                    print(template.format(self.describe[l].get(i)), end='')
                print()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process some integers.')
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--file', nargs=1,
                            help="train csv file is dataset with feature, default is datasets/dataset_train.csv",
                            default=['data/data.csv'], required=False)
        parser.add_argument('--delimiter', nargs=1,
                            help="train csv file is dataset with feature, default is datasets/dataset_train.csv",
                            default=[','], required=False)
        ds = DataSet(parser.parse_args().file[0], delim=parser.parse_args().delimiter[0])
        ds.find_numeric_label()
        ds.output_describe()
    except Exception:
        print('Error with arguments')