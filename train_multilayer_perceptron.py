import numpy as np
from abc import ABC
import pandas as pd
import json
import sys
from matplotlib import pyplot as plt
import argparse

import torch
from torch import nn
from torch.autograd import Variable

class Preprocess(ABC):
    def __init__(self, x):
        self.x = x

    def preprocessing_std(self):
        self.x = (self.x - self.x.mean(axis=0)) / self.x.std(axis=0)
        # for i in range(len(self.x)):
        #     self.x[i] = ((self.x[i] - self.x[i].mean()) / self.x[i].std()) # переписать на numpy

    def preprocessing_min_max(self):
        for i in range(len(mlp.x)):
            self.x[i] = (self.x[i] - self.x[i].min()) / (self.x[i].max() - self.x[i].min())

    def preprocessing_max(self):
        for i in range(len(mlp.x)):
            self.x[i] = self.x[i] / self.x[i].max()


EPSILON = 1e-7


class BCE():
    def binary_cross_entropy(self, output, target):
        # вывести значение loss
        output = np.clip(output, EPSILON, 1 - EPSILON)
        res = -(np.log(np.array([i if j == 1 else 1 - i for i, j in zip(output, target)])).mean())
        return (res)

    def gradient_binary_cross_entropy(self, output, target):
        output = np.clip(output, EPSILON, 1 - EPSILON)
        return np.where(target == 1, -1 / output, 1 / (1 - output)) / output.shape[0]

class Test(BCE):
    def test_loss(self, x, y):
        loss = torch.nn.BCELoss()
        inp = torch.from_numpy(x)
        y = torch.from_numpy(y)
        y_pred_gt = inp.detach().clone()
        y_pred_gt.requires_grad = True
        gt = loss(y_pred_gt, y)
        gt.backward()
        print ("distantion of Loss {}".format(torch.norm(gt - torch.from_numpy(np.array(self.binary_cross_entropy(inp.reshape(-1), y.reshape(-1)))),
                   dim=-1).sum()))
        print ("distantion of gradient Loss {}".torch.norm(y_pred_gt.grad.reshape(-1) - torch.from_numpy(
            self.gradient_binary_cross_entropy(inp.reshape(-1), y.reshape(-1)))))


class Multilayer_Perceptron(Preprocess, BCE):
    def __init__(self, file='data/data.csv', epohs=2, batch_size=0, lr=0.9):
        self.df = pd.read_csv(file)
        self.x = np.array(self.df[self.df.columns[2:]])
        self.y = np.array(self.df[self.df.columns[1]])
        self.batch_size = batch_size
        if (batch_size <= 0 or batch_size > self.x.shape[0]):
            self.batch_size = self.x.shape[0]
        self.epohs = epohs
        self.lr = lr
        if (lr < 0 or lr > 1):
            self.lr = 0.15

        self.plot_dict = {'plot_x_test' : [], 'plot_x_train' : [], 'plot_epoh' : [],
                          'plot_accuracy': [], 'plot_precicion': [], 'plot_recall': [], 'plot_recall': [],  'plot_f1': [],
                          'plot_accuracy_test' : [], 'plot_precicion_test' : [], 'plot_recall_test' : [], 'plot_recall_test' : [], 'plot_f1_test' : []
                          }


    def accuracy_metric(self, output, target):
        self.tp = output[target == 1]
        self.fn = self.tp[self.tp <= 0.5].shape[0]
        self.tp = self.tp.shape[0] - self.fn
        self.fp = output[target != 1]
        self.tn = self.fp[self.fp <= 0.5].shape[0]
        self.fp = self.fp.shape[0] - self.tn
        return ((self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn))

    def precision_metric(self, output, target, flag=True):
        if flag:
            self.accuracy_metric(output, target)
        self.precision = self.tp / (self.tp + self.fp)
        return (self.precision)

    def recall_metric(self, output, target, flag=True):
        if flag:
            self.accuracy_metric(output, target)
        self.recall = self.tp / (self.tp + self.fn)
        return (self.recall)

    def Fn_metric(self, output, target, n=1, flag=True):
        if flag:
            self.accuracy_metric(output, target)
            self.precision_metric(output, target, flag=False)
            self.recall_metric(output, target, flag=False)
        return (2.0 * ((self.precision * self.recall) / ((self.precision) + self.recall)))

    def soft_max(self, x):
        res = np.array([np.exp(x[0]) / np.exp(x).sum(), np.exp(x[1]) / np.exp(x).sum()])
        return (res)

    def sigmoid(self, x):
        return (1.0 / (1.0 + np.exp(-x)))

    def Relu(self, x):
        x[x < 0] = 0
        return x
        # return (np.array([0 if i < 0 else i for j in x for i in j]).reshape(x.shape))


    def add_intercept(self, x):
        intercept = np.ones((x.shape[0], 1))
        return (np.concatenate((intercept, x), axis=1))

    def init_weight(self, height, width):
        return (np.random.randn(height, width) * np.sqrt(2 / width))

    def backpropagation(self, X, y):

        grad_loss = self.gradient_binary_cross_entropy(self.output_neurons, y)

        dA = (grad_loss * self.output_neurons).sum(axis=1).reshape(-1, 1)
        grad_w3 = self.output_neurons * (grad_loss - dA) / 2

        d_W3 = grad_w3.T @ self.neurons_H2
        d_b3 = grad_w3.sum(axis=0)
        grad_w2 = grad_w3 @ self.weights_H2toO

        grad_w2[self.neurons_H2 == 0] = 0
        d_W2 = grad_w2.T @ self.neurons_H1
        d_b2 = grad_w2.sum(axis=0)
        grad_w1 = grad_w2 @ self.weights_H1toH2

        grad_w1[self.neurons_H1 == 0] = 0
        d_W1 = grad_w1.T @ X
        d_b1 = grad_w1.sum(axis=0)

        self.weights_ItoH1 -= self.lr * d_W1
        self.bias_TtoH1 -= self.lr * d_b1.reshape(1, -1)
        self.weights_H1toH2 -= self.lr * d_W2
        self.bias_H1toH2 -= self.lr * d_b2.reshape(1, -1)
        self.weights_H2toO -= self.lr * d_W3
        self.bias_H2toO -= self.lr * d_b3.reshape(1, -1)

    def feedforward(self, X):
        self.neurons_H1 = np.dot(X, self.weights_ItoH1.T) + self.bias_TtoH1
        self.neurons_H1 = self.Relu(self.neurons_H1)

        self.neurons_H2 = np.dot(self.neurons_H1, self.weights_H1toH2.T) + self.bias_H1toH2
        self.neurons_H2 = self.Relu(self.neurons_H2)

        self.output_neurons = np.dot(self.neurons_H2, self.weights_H2toO.T) + self.bias_H2toO
        self.output_neurons = np.array([self.soft_max(i) for i in self.output_neurons])

    def train_test_split(self):
        mas_index = np.array(range(self.x.shape[0]))
        np.random.shuffle(mas_index)

        sample = self.x[mas_index]
        target = np.array([(mlp.y == 'M').astype(int), (mlp.y != 'M').astype(int)]).T[mas_index]

        s = int(sample.shape[0] * 0.1)
        if s == 0:
            s = 1

        x_test = np.array(sample[:s])
        y_test = np.array(target[:s])

        x_train = np.array(sample[s:])
        y_train = np.array(target[s:])

        return (x_train, x_test, y_train, y_test)

    def train_split(self):
        mas_index = np.array(range(self.x.shape[0]))
        np.random.shuffle(mas_index)

        sample = self.x[mas_index]
        target = np.array([(mlp.y == 'M').astype(int), (mlp.y != 'M').astype(int)]).T[mas_index]
        return sample, target


    def shuffle_dataset(self):
        mas_index = np.array(range(self.x_test.shape[0]))
        np.random.shuffle(mas_index)
        self.x_test = self.x_test[mas_index]
        self.y_test = self.y_test[mas_index]

        mas_index = np.array(range(self.x_val.shape[0]))
        np.random.shuffle(mas_index)
        self.x_val = self.x_val[mas_index]
        self.y_val = self.y_val[mas_index]

    def batch_iterator(self, X, y, permute):
        if permute:
            mas_index = np.array(range(X.shape[0]))
            np.random.shuffle(mas_index)
            X = X[mas_index]
            y = y[mas_index]
        for i in range(0, X.shape[0], self.batch_size):
            begin, end = i, min(i + self.batch_size, X.shape[0])
            yield X[begin:end], y[begin:end]

    def save_weight(self):
        frame = {'weights_ItoH1': self.weights_ItoH1.tolist(),
                 'weights_H1toH2': self.weights_H1toH2.tolist(),
                 'weights_H2toO': self.weights_H2toO.tolist(),
                 'bias_TtoH1': self.bias_TtoH1.tolist(),
                 'bias_H1toH2': self.bias_H1toH2.tolist(),
                 'bias_H2toO': self.bias_H2toO.tolist()}
        with open('weight_mlp.json', 'w', encoding='utf-8') as file:
            json.dump(frame, file)

    def save_epoh_loss(self, epoh, flag_test, flag_metrics=False):
        self.feedforward(self.x_train)
        res_train_loss = self.output_neurons[:, 1]
        res_accuracy = self.accuracy_metric(res_train_loss, self.y_train[:, 1])
        res_precision = self.precision_metric(res_train_loss, self.y_train[:, 1], flag=False)
        res_recall = self.recall_metric(res_train_loss, self.y_train[:, 1], flag=False)
        res_f1 = self.Fn_metric(res_train_loss, self.y_train[:, 1], flag=False)
        self.plot_dict['plot_accuracy'].append(res_accuracy)
        self.plot_dict['plot_precicion'].append(res_precision)
        self.plot_dict['plot_recall'].append(res_recall)
        self.plot_dict['plot_f1'].append(res_f1)
        self.plot_dict['plot_x_train'].append(self.binary_cross_entropy(res_train_loss, self.y_train[:, 1]))

        if flag_test:
            self.feedforward(self.x_test)
            res_loss_test = self.output_neurons[:, 1]

            res_accuracy_test = self.accuracy_metric(res_train_loss, self.y_train[:, 1])
            res_precision_test = self.precision_metric(res_train_loss, self.y_train[:, 1], flag=False)
            res_recall_test = self.recall_metric(res_train_loss, self.y_train[:, 1], flag=False)
            res_f1_test = self.Fn_metric(res_train_loss, self.y_train[:, 1], flag=False)
            self.plot_dict['plot_accuracy_test'].append(res_accuracy_test)
            self.plot_dict['plot_precicion_test'].append(res_precision_test)
            self.plot_dict['plot_recall_test'].append(res_recall_test)
            self.plot_dict['plot_f1_test'].append(res_f1_test)

            self.plot_dict['plot_x_test'].append(self.binary_cross_entropy(res_loss_test, self.y_test[:, 1]))
            print('epoch {}/{} - loss: {} - val_loss: {} - f1: {}'.format(epoh + 1, self.epohs,
                                                                 self.binary_cross_entropy(res_train_loss,
                                                                                           self.y_train[:, 1]),
                                                                 self.binary_cross_entropy(res_loss_test,
                                                                                           self.y_test[:, 1]),
                                                                res_f1_test))
            if flag_metrics:
                print('accuracy: {} - precision: {} - recall: {}'.format(res_accuracy, res_precision,
                                                                         res_recall))
                print('accuracy_test: {} - precision_test: {} - recall_test: {}'.format(res_accuracy_test, res_precision_test,
                                                                         res_recall_test))
        else:
            print('epoch {}/{} - loss: {} - f1: {}'.format(epoh + 1, self.epohs,
                                                  self.binary_cross_entropy(res_train_loss,
                                                                            self.y_train[:, 1]),
                                                           res_f1))
            if flag_metrics:
                print('accuracy: {} - precision: {} - recall: {}'.format(res_accuracy, res_precision,
                                                                         res_recall))


    def built_plot(self, flag_test=False):
        plt.plot(self.plot_dict['plot_epoh'], self.plot_dict['plot_x_train'], 'r', label="loss_train", color='r')
        plt.plot(self.plot_dict['plot_epoh'], self.plot_dict['plot_accuracy'], 'r', label="accuracy ", color='b')
        plt.plot(self.plot_dict['plot_epoh'], self.plot_dict['plot_precicion'], 'r', label="precision", color='g')
        plt.plot(self.plot_dict['plot_epoh'], self.plot_dict['plot_recall'], 'r', label="recall", color='c')
        plt.plot(self.plot_dict['plot_epoh'], self.plot_dict['plot_f1'], 'r', label="f1", color='m')

        if flag_test:
            plt.plot(self.plot_dict['plot_epoh'], self.plot_dict['plot_x_test'], 'r', label="loss_test", color='#f97306')
            plt.plot(self.plot_dict['plot_epoh'], self.plot_dict['plot_accuracy_test'], 'r', label="accuracy_test", color='y')
            plt.plot(self.plot_dict['plot_epoh'], self.plot_dict['plot_precicion_test'], 'r', label="precision_test", color='k')
            plt.plot(self.plot_dict['plot_epoh'], self.plot_dict['plot_recall_test'], 'r', label="recall_test", color='#d6b4fc')
            plt.plot(self.plot_dict['plot_epoh'], self.plot_dict['plot_f1_test'], 'r', label="f1_test", color='#d648d7')


        plt.title('Plot loss and metrics!')
        plt.legend(loc='lower right')
        plt.xlabel('epoh', fontsize=14, fontweight='bold')
        plt.ylabel('y', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=10, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        plt.savefig('data/pair_plot.png')
        plt.show()

    def train(self, h1_dim=20, h2_dim=10, flag_test=False, flag_metrics=False):
        self.preprocessing_std()

        self.weights_ItoH1 = self.init_weight(h1_dim, self.x.shape[1])
        self.bias_TtoH1 = self.init_weight(1, h1_dim)

        self.weights_H1toH2 = self.init_weight(h2_dim, h1_dim)
        self.bias_H1toH2 = self.init_weight(1, h2_dim)

        self.weights_H2toO = self.init_weight(2, h2_dim)
        self.bias_H2toO = self.init_weight(1, 2)

        # if flag_test:
        #     self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split()
        # else:
        #     self.x_train, self.y_train = self.train_split()

        for epoh in range(self.epohs):
            self.plot_dict['plot_epoh'].append(epoh)

            if flag_test:
                self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split()
            else:
                self.x_train, self.y_train = self.train_split()

            for batch_x, batch_y in self.batch_iterator(self.x_train, self.y_train, True):

                # feedforward
                self.feedforward(batch_x)
                # backpropogation
                self.backpropagation(batch_x, batch_y)

            self.save_epoh_loss(epoh, flag_test, flag_metrics)
        self.save_weight()
        self.built_plot(flag_test=flag_test)


    def predict(self, flag_metrics=False):
        self.preprocessing_std()
        with open('weight_mlp.json', 'r', encoding='utf-8') as file:
            frame = json.load(file)
        self.weights_ItoH1 = np.array(frame['weights_ItoH1'])
        self.weights_H1toH2 = np.array(frame['weights_H1toH2'])
        self.weights_H2toO = np.array(frame['weights_H2toO'])
        self.bias_TtoH1 = np.array(frame['bias_TtoH1'])
        self.bias_H1toH2 = np.array(frame['bias_H1toH2'])
        self.bias_H2toO = np.array(frame['bias_H2toO'])

        self.epohs = 1
        self.x_train, self.y_train = self.train_split()
        self.feedforward(self.x_train)

        print('#####################################')
        print('finish resultat:')
        self.save_epoh_loss(epoh=0, flag_test=False, flag_metrics=flag_metrics)
        print('#####################################')



if  __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', nargs=1,
                        help="input train or test file, default is data.csv",
                        default=['data/data_training.csv'], required=False)
    parser.add_argument('--status', nargs=1,
                        help="It's status programm, input 't' or 'p' it is train or predict, default value is t (train)",
                        default=['t'], required=False)
    parser.add_argument('--batch_size', nargs=1,
                        help="input value for batch_size",
                        default=[100], type=int, required=False)
    parser.add_argument('--epohs', nargs=1,
                        help="input value for batch_size",
                        default=[30], type=int, required=False)
    parser.add_argument('--lr', nargs=1,
                        help="Input value learning rate, default value is 0.15",
                        default=[0.15], type=float, required=False)
    parser.add_argument('--flag_test', nargs=1,
                        help="It's status programm, input 't' or 'p' it is train or predict, default value is t (train)",
                        default=[1], type=int, required=False)
    parser.add_argument('--flag_metrics', nargs=1,
                        help='input 0 or 1 for output metrics accuracy, precision and recall',
                        default=[0], type=int, required=False)

    pars = parser.parse_args()

    if pars.status[0] == 't':
        mlp = Multilayer_Perceptron(file=pars.file[0], batch_size=pars.batch_size[0], epohs=pars.epohs[0], lr=pars.lr[0])
        mlp.train(flag_test=bool(pars.flag_test[0]), flag_metrics=bool(pars.flag_metrics[0]))
    elif pars.status[0] == 'p':
        mlp = Multilayer_Perceptron(file=pars.file[0], batch_size=pars.batch_size[0], epohs=pars.epohs[0], lr=pars.lr[0])
        mlp.predict(flag_metrics=bool(pars.flag_metrics[0]))