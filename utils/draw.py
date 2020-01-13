# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 9:58 上午
# @Author  : zhangpin
# @Email   : zhangpin@geetest.com
# @File    : draw
# @Software: PyCharm

import os
from parser import parameter_parser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, p

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来显示中文字体
plt.rcParams["font.serif"] = ["Times New Roman"]  # 用来显示西文字体
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


class Ratio(object):
    def __init__(self, output_root, layer, gcn_dim, hour, args):

        self.hour = hour
        # train set: 19-09-01
        self.train_01_layer_dim_df = pd.read_csv( \
            os.path.join(output_root, "logs/train_01_layer_{}_dim_{}.csv").format(layer, gcn_dim))
        # train set: 19-09-01 hour: hour
        index = [min(idx + self.hour, self.train_01_layer_dim_df.shape[0]) for idx in self.train_01_layer_dim_df.index if not idx % 24]
        self.train_01_layer_dim_hour_df = self.train_01_layer_dim_df.iloc[index, :]
        # test set: 19-09-02/19-09-05
        self.test_02_layer_dim_df = pd.read_csv( \
            os.path.join(output_root, "logs/test_02_layer_{}_dim_{}.csv").format(layer, gcn_dim))
        self.test_05_layer_dim_df = pd.read_csv( \
            os.path.join(output_root, "logs/test_05_layer_{}_dim_{}.csv").format(layer, gcn_dim))

        self.output_root = output_root
        self.args = args
        self.layer = layer
        self.dim = gcn_dim

    def draw_ratio(self, key):
        """
        draw ratio: loss, acc, precision_recall, macro_f1, micro_f1
        @param key: draw which ratio
        """

        if key == "loss":
            # 19-09-01整天的损失对比
            #             plt.figure()
            #             plt.plot(np.arange(self.train_01_layer_dim_df.shape[0]), self.train_01_layer_dim_df[key])
            #             plt.xlabel("epoch")
            #             plt.ylabel("loss")
            #             plt.title("19-09-01整天的损失(层数{}维度{})".format(self.layer, self.dim))
            #             # plt.legend(labels=["19-09-01"])
            #             plt.savefig(os.path.join(self.output_root, "imgs/train_01_layer_{}_dim_{}_loss.png" \
            #                                      .format(self.layer, self.dim)), dpi=self.args.dpi)

            # 19-09-01当天上第*个小时的损失对比
            plt.figure()
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]), self.train_01_layer_dim_hour_df[key], linewidth=1)
            plt.title("19-09-01第{}个小时的{}(层数{}维度{})".format(self.hour, key, self.layer, self.dim))
            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            plt.xlim(-0.1, args.epochs + 0.1)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.savefig(os.path.join(self.output_root, "imgs/loss/train_01_hour_{}_layer_{}_dim_{}_loss.png" \
                                     .format(self.hour, self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

        if key == "acc":
            # 测试集上没有损失
            # 19-09-02/19-09-05整天的准确率对比
            plt.figure()
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]), self.test_02_layer_dim_df[key], "teal", linewidth=1)
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]), self.test_05_layer_dim_df[key], "olivedrab", linewidth=1)
            plt.title("19-09-02/19-09-05每小时的准确率(层数{}维度{})".format(self.layer, self.dim))
            plt.legend(loc="best", labels=["19-09-02", "19-09-05"])
            ax = plt.gca()
            plt.xlim(-0.1, 23.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
            plt.xlabel("hour")
            plt.ylabel("acc")
            plt.savefig(os.path.join(self.output_root, "imgs/acc/test_02_05_layer_{}_dim_{}_acc.png" \
                                     .format(self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

            # 19-09-01第*个小时的准确率对比
            plt.figure()
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]), self.train_01_layer_dim_hour_df[key], "teal")
            plt.title("19-09-01第{}个小时的准确率(层数{}维度{})".format(self.hour, self.layer, self.dim))
            ax = plt.gca()
            plt.xlim(-0.1, args.epochs + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("epoch")
            plt.ylabel("acc")
            plt.savefig(os.path.join(self.output_root, "imgs/acc/train_01_hour_{}_layer_{}_dim_{}_acc.png" \
                                     .format(self.hour, self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

        if key == "white":
            # 训练集上的白样本精准率
            plt.figure()
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["white_p"])
            plt.title("19-09-01第{}小时白样本的精准率(层数{}维度{})".format(self.hour, self.layer, self.dim))
            ax = plt.gca()
            plt.xlim(-0.1, args.epochs + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("epoch")
            plt.ylabel("precision")
            plt.savefig(os.path.join(self.output_root, "imgs/white_p/train_01_hour_{}_layer_{}_dim_{}_white_p.png" \
                                     .format(self.hour, self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

            # 测试集上的白样本精准率
            plt.figure()
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["white_p"])
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["white_p"])
            plt.title("19-09-02/19-09-05白样本的精准率(层数{}维度{})".format(self.layer, self.dim))
            plt.legend(loc="best", labels=["19-09-02", "19-09-05"])
            ax = plt.gca()
            plt.xlim(-0.1, 23.5)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("hour")
            plt.ylabel("precision")
            plt.savefig(os.path.join(self.output_root, "imgs/white_p/test_02_05_layer_{}_dim_{}_white_p.png" \
                                     .format(self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

            # 训练集上的白样本召回率
            plt.figure()
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["white_r"])
            plt.title("19-09-01第{}小时白样本的召回率(层数{}维度{})".format(self.hour, self.layer, self.dim))
            ax = plt.gca()
            plt.xlim(-0.1, args.epochs + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("epoch")
            plt.ylabel("recall")
            plt.savefig(os.path.join(self.output_root, "imgs/white_r/train_01_hour_{}_layer_{}_dim_{}_white_r.png" \
                                     .format(self.hour, self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

            # 测试集上的白样本召回率
            plt.figure()
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["white_r"])
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["white_r"])
            plt.title("19-09-02/19-09-05白样本的召回率(层数{}维度{})".format(self.layer, self.dim))
            plt.legend(loc="best", labels=["19-09-02", "19-09-05"])
            ax = plt.gca()
            plt.xlim(-0.1, 23.5)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("hour")
            plt.ylabel("recall")
            plt.savefig(os.path.join(self.output_root, "imgs/white_r/test_02_05_layer_{}_dim_{}_white_r.png" \
                                     .format(self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

            # 训练集19-09-01第*小时的白样本的精准与召回
            plt.figure()
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["white_p"], "orange", linestyle="-")
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["white_r"], color="orange", linestyle="--")
            plt.title("19-09-01第{}个小时白样本的精准与召回率(层数{}维度{})".format(hour + 1, self.layer, self.dim))
            plt.legend(loc="best", labels=["precision", "recall"])
            ax = plt.gca()
            plt.xlim(-0.1, args.epochs + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("epoch")
            plt.savefig(os.path.join(self.output_root, "imgs/white_pr/train_01_hour_{}_layer_{}_dim_{}_white_pr.png") \
                        .format(self.hour, self.layer, self.dim), dpi=self.args.dpi)
            plt.close()

            # 测试集19-09-02白样本的精准与召回
            plt.figure()
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["white_p"], "orange", linestyle="-")
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["white_r"], "orange", linestyle="--")
            plt.title("19-09-02白样本的精准与召回率(层数{}维度{})".format(self.layer, self.dim))
            plt.legend(loc="best", labels=["precision", "recall"])
            ax = plt.gca()
            plt.xlim(-0.1, 23 + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("hour")
            #             plt.ylabel("precision_recall")
            plt.savefig(os.path.join(self.output_root, "imgs/white_pr/test_02_layer_{}_dim_{}_white_pr.png") \
                        .format(self.layer, self.dim), dpi=self.args.dpi)
            plt.close()

            # 测试集19-09-05上白样本的精准与召回
            plt.figure()
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["white_p"], "orange", linestyle="-")
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["white_r"], "orange", linestyle="--")
            plt.title("19-09-05白样本的精准与召回率(层数{}维度{})".format(self.layer, self.dim))
            plt.legend(loc="best", labels=["precision", "recall"])
            ax = plt.gca()
            plt.xlim(-0.1, 23 + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("hour")
            #             plt.ylabel("precision_recall")
            plt.savefig(os.path.join(self.output_root, "imgs/white_pr/test_05_layer_{}_dim_{}_white_pr.png") \
                        .format(self.layer, self.dim), dpi=self.args.dpi)
            plt.close()

        if key == "black":
            # 训练集上的黑样本精准率
            plt.figure()
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["black_p"])
            plt.title("19-09-01黑样本的精准率(层数{}维度{})".format(self.layer, self.dim))
            ax = plt.gca()
            plt.xlim(-0.1, args.epochs + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("epoch")
            plt.ylabel("precision")
            plt.savefig(os.path.join(self.output_root, "imgs/black_p/train_01_hour_{}_layer_{}_dim_{}_black_p.png" \
                                     .format(self.hour, self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

            # 测试集上的黑样本精准率
            plt.figure()
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["black_p"])
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["black_p"])
            plt.title("19-09-02/19-09-05黑样本的精准率(层数{}维度{})".format(self.layer, self.dim))
            plt.legend(loc="best", labels=["19-09-02", "19-09-05"])
            ax = plt.gca()
            plt.xlim(-0.1, 23 + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("hour")
            plt.ylabel("precision")
            plt.savefig(os.path.join(self.output_root, "imgs/black_p/test_02_05_layer_{}_dim_{}_black_p.png" \
                                     .format(self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

            # 训练集上的黑样本召回率
            plt.figure()
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["black_r"])
            plt.title("19-09-01黑样本的召回率(层数{}维度{})".format(self.layer, self.dim))
            ax = plt.gca()
            plt.xlim(-0.1, args.epochs + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("epoch")
            plt.ylabel("recall")
            plt.savefig(os.path.join(self.output_root, "imgs/black_r/train_01_hour_{}_layer_{}_dim_{}_black_r.png" \
                                     .format(self.hour, self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

            # 测试集上的黑样本召回率
            plt.figure()
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["black_r"])
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["black_r"])
            plt.title("19-09-02/19-09-05黑样本的召回率(层数{}维度{})".format(self.layer, self.dim))
            plt.legend(loc="best", labels=["19-09-02", "19-09-05"])
            ax = plt.gca()
            plt.xlim(-0.1, 23 + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("hour")
            plt.ylabel("precision")
            plt.savefig(os.path.join(self.output_root, "imgs/black_r/test_02_05_layer_{}_dim_{}_black_r.png" \
                                     .format(self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

            # 19-09-01当天第*小时的黑样本的精准和召回率
            plt.figure()
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["black_p"], "c-")
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["black_r"], "c--")
            plt.title("19-09-01第{}个小时白样本的精准与召回率(层数{}维度{})".format(hour + 1, self.layer, self.dim))
            plt.legend(loc="best", labels=["precision", "recall"])
            ax = plt.gca()
            plt.xlim(-0.1, args.epochs + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("epoch")
            #             plt.ylabel("precision_recall")
            plt.savefig(os.path.join(self.output_root, "imgs/black_pr/train_01_hour_{}_layer_{}_dim_{}_black_pr.png") \
                        .format(self.hour, self.layer, self.dim), dpi=self.args.dpi)
            plt.close()

            # 测试集19-09-02上黑样本的精准与召回
            plt.figure()
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["black_p"], "c-")
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["black_r"], "c--")
            plt.title("19-09-02黑样本的精准与召回率(层数{}维度{})".format(self.layer, self.dim))
            plt.legend(loc="best", labels=["precision", "recall"])
            ax = plt.gca()
            plt.xlim(-0.1, 23 + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("hour")
            #             plt.ylabel("precision_recall")
            plt.savefig(os.path.join(self.output_root, "imgs/black_pr/test_02_layer_{}_dim_{}_black_pr.png") \
                        .format(self.layer, self.dim), dpi=self.args.dpi)
            plt.close()

            # 测试集19-09-05上黑样本的精准与召回
            plt.figure()
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["black_p"], "b-")
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["black_r"], "b--")
            plt.title("19-09-05黑样本的精准与召回率(层数{}维度{})".format(self.layer, self.dim))
            plt.legend(loc="best", labels=["precision", "recall"])
            ax = plt.gca()
            plt.xlim(-0.1, 23 + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("hour")
            plt.savefig(os.path.join(self.output_root, "imgs/black_pr/test_05_layer_{}_dim_{}_black_pr.png") \
                        .format(self.layer, self.dim), dpi=self.args.dpi)
            plt.close()

        if key == "precision_recall":
            # 19-09-01 当天第*小时黑白样本的精准和召回
            plt.figure()
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["white_p"], "orange", linestyle="-")
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["white_r"], "orange", linestyle="--")
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["black_p"], "c-")
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["black_r"], "c--")
            plt.title("19-09-01第{}小时黑白样本的精准和召回率(层数{}维度{})".format(self.hour, self.layer, self.dim))
            plt.legend(loc="best", labels=["white_p", "white_r", "black_p", "black_r"])
            ax = plt.gca()
            plt.xlim(-0.1, args.epochs + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("epoch")
            plt.savefig(os.path.join(self.output_root, \
                                     "imgs/white_black_pr/train_01_hour_{}_layer_{}_dim_{}_white_black_pr.png") \
                        .format(self.hour, self.layer, self.dim), dpi=self.args.dpi)
            plt.close()

            # 19-09-02 当天黑白样本的精准和召回
            plt.figure()
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["white_p"], "orange", linestyle="-")
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["white_r"], "orange", linestyle="--")
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["black_p"], "c-")
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["black_r"], "c--")
            plt.title("19-09-02黑白样本的精准和召回率(层数{}维度{})".format(self.layer, self.dim))
            plt.legend(loc="best", labels=["white_p", "white_r", "black_p", "black_r"])
            ax = plt.gca()
            plt.xlim(-0.1, 23 + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("hour")
            plt.savefig(os.path.join(self.output_root, "imgs/white_black_pr/test_02_layer_{}_dim_{}_white_black_pr.png" \
                                     .format(self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

            # 19-09-05 当天黑白样本的精准和召回
            plt.figure()
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["white_p"], "orange", linestyle="-")
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["white_r"], "orange", linestyle="--")
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["black_p"], "c-")
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["black_r"], "c--")
            plt.title("19-09-05黑白样本的精准和召回率(层数{}维度{})".format(self.layer, self.dim))
            plt.legend(loc="best", labels=["white_p", "white_r", "black_p", "black_r"])
            ax = plt.gca()
            plt.xlim(-0.1, 23 + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("hour")
            plt.savefig(os.path.join(self.output_root, "imgs/white_black_pr/test_05_layer_{}_dim_{}_white_black_pr.png") \
                        .format(self.layer, self.dim), dpi=self.args.dpi)
            plt.close()

        if key == "macro_f1":
            # 训练集第*小时macro_f1
            plt.figure()
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["macro_f1"])
            plt.title("19-09-01第{}小时的macro_f1(层数{}维度{})".format(self.hour, self.layer, self.dim))
            ax = plt.gca()
            plt.xlim(-0.1, args.epochs + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.savefig(os.path.join(self.output_root, "imgs/f1_score/train_01_hour_{}_layer_{}_dim_{}_macro_f1.png" \
                                     .format(self.hour, self.layer, self.dim)), dpi=self.args.dpi)
            plt.xlabel("epoch")
            plt.ylabel("macro-f1")
            plt.close()

            # 测试集上的macro_f1
            plt.figure()
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["macro_f1"])
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["macro_f1"])
            plt.title("19-09-02/19-09-05的macro_f1(层数{}维度{})".format(self.layer, self.dim))
            plt.legend(loc="best", labels=["19-09-02", "19-09-05"])
            ax = plt.gca()
            plt.xlim(-0.1, 23 + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("hour")
            plt.ylabel("macro-f1")
            plt.savefig(os.path.join(self.output_root, "imgs/f1_score/test_02_05_layer_{}_dim_{}_macro_f1.png" \
                                     .format(self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

        if key == "micro_f1":
            # 训练集的micro_f1
            plt.figure()
            plt.plot(np.arange(self.train_01_layer_dim_hour_df.shape[0]),
                     self.train_01_layer_dim_hour_df["micro_f1"])
            plt.title("19-09-01第{}小时的micro_f1(层数{}维度{})".format(self.hour, self.layer, self.dim))
            ax = plt.gca()
            plt.xlim(-0.1, args.epochs + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("epoch")
            plt.ylabel("micro-f1")
            plt.savefig(os.path.join(self.output_root, "imgs/f1_score/train_01_hour_{}_layer_{}_dim_{}_micro_f1.png" \
                                     .format(self.hour, self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()

            # 测试集上的micro_f1
            plt.figure()
            plt.plot(np.arange(self.test_02_layer_dim_df.shape[0]),
                     self.test_02_layer_dim_df["micro_f1"])
            plt.plot(np.arange(self.test_05_layer_dim_df.shape[0]),
                     self.test_05_layer_dim_df["micro_f1"])
            plt.title("19-09-02/19-09-05的micro_f1(层数{}维度{})".format(self.layer, self.dim))
            plt.legend(loc="best", labels=["19-09-02", "19-09-05"])
            ax = plt.gca()
            plt.xlim(-0.1, 23 + 0.1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            plt.ylim(0.85, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
            plt.xlabel("hour")
            plt.ylabel("micro-f1")
            plt.savefig(os.path.join(self.output_root, "imgs/f1_score/test_02_05_layer_{}_dim_{}_micro_f1.png" \
                                     .format(self.layer, self.dim)), dpi=self.args.dpi)
            plt.close()


class PRecall():
    def __init__(self, root_dir, layer, dim, hour, args):
        self.layer = layer
        self.dim = dim
        self.hour = hour
        self.args = args
        self.root_dir = root_dir

        self.test_02_hour_layer_dim = pd.read_csv( \
            os.path.join(self.root_dir, "curves/test_02_hour_{}_layer_{}_dim_{}_pr.csv" \
                         .format(self.hour, self.layer, self.dim)))
        self.test_05_hour_layer_dim = pd.read_csv( \
            os.path.join(self.root_dir, "curves/test_05_hour_{}_layer_{}_dim_{}_pr.csv" \
                         .format(self.hour, self.layer, self.dim)))

    def drawPR(self):
        """
        draw P-R curve
        """
        # 19-09-02第*小时的P-R曲线
        label = self.test_02_hour_layer_dim["label"]
        prob = []
        for i in range(label.shape[0]):
            if label[i] == 0:
                prob.append(self.test_02_hour_layer_dim["neg_prob"][i])
            else:
                prob.append(self.test_02_hour_layer_dim["pos_prob"][i])
        prob = np.array(prob)
        precision, recall, threshold = precision_recall_curve(label, prob)
        plt.plot(precision, recall, color="#20B2AA", linewidth=2)
        plt.title("19-09-02第{}小时的P-R曲线(层数{}维度{})".format(self.hour, self.layer, self.dim))
        ax = plt.gca()
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(self.root_dir, "imgs/curves/test_02_hour_{}_layer_{}_dim_{}_pr.png" \
                                 .format(self.hour, self.layer, self.dim)), dpi=self.args.dpi)
        plt.close()

        # 19-09-05第*小时的P-R曲线
        label = self.test_05_hour_layer_dim["label"]
        prob = []
        for i in range(label.shape[0]):
            if label[i] == 0:
                prob.append(self.test_05_hour_layer_dim["neg_prob"][i])
            else:
                prob.append(self.test_05_hour_layer_dim["pos_prob"][i])
        prob = np.array(prob)
        precision, recall, threshold = precision_recall_curve(label, prob)
        plt.plot(precision, recall, color="#20B2AA", linewidth=2)
        plt.title("19-09-05第{}小时的P-R曲线(层数{}维度{})".format(self.hour, self.layer, self.dim))
        #         plt.xlim(0.0, 1.0)
        #         plt.ylim(0.0, 1.0)
        ax = plt.gca()
        #         ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        #         ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(self.root_dir, "imgs/curves/test_05_hour_{}_layer_{}_dim_{}_pr.png" \
                                 .format(self.hour, self.layer, self.dim)), dpi=self.args.dpi)
        plt.close()

    def drawROC(self):
        # 19-09-02第*小时的ROC曲线
        label = self.test_02_hour_layer_dim["label"]
        prob = []
        for i in range(label.shape[0]):
            if label[i] == 0:
                prob.append(self.test_02_hour_layer_dim["neg_prob"][i])
            else:
                prob.append(self.test_02_hour_layer_dim["pos_prob"][i])
        prob = np.array(prob)
        precision, recall, threshold = roc_curve(label, prob, pos_label=1)
        plt.plot(precision, recall, color="#FF8C00", linewidth=2)
        plt.title("19-09-02第{}小时的ROC曲线(层数{}维度{})".format(self.hour, self.layer, self.dim))
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        plt.xlabel("假阳性率")
        plt.ylabel("真阳性率")
        plt.legend(loc="best", labels=["{:.4f}".format(roc_auc_score(label, prob))], \
                   frameon=True, framealpha=0.25, title="AUC")
        plt.savefig(os.path.join(self.root_dir, "imgs/curves/test_02_hour_{}_layer_{}_dim_{}_roc.png" \
                                 .format(self.hour, self.layer, self.dim)), dpi=self.args.dpi)
        plt.close()

        # 19-09-05第*小时的ROC曲线
        label = self.test_05_hour_layer_dim["label"]
        prob = []
        for i in range(label.shape[0]):
            if label[i] == 0:
                prob.append(self.test_05_hour_layer_dim["neg_prob"][i])
            else:
                prob.append(self.test_05_hour_layer_dim["pos_prob"][i])
        prob = np.array(prob)
        precision, recall, threshold = roc_curve(label, prob, pos_label=1)
        plt.plot(precision, recall, color="#FF8C00", linewidth=2)
        plt.title("19-09-05第{}小时的ROC曲线(层数{}维度{})".format(self.hour, self.layer, self.dim))
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        plt.xlabel("假阳性率")
        plt.ylabel("真阳性率")
        plt.legend(loc="best", labels=["AUC={:.4f}".format(roc_auc_score(label, prob))])
        plt.savefig(os.path.join(self.root_dir, "imgs/curves/test_05_hour_{}_layer_{}_dim_{}_roc.png" \
                                 .format(self.hour, self.layer, self.dim)), dpi=self.args.dpi)
        plt.close()


class LayerDim(object):
    def __init__(self, output_root, output_dirs, hour, args):
        """
        @param output_dirs: list[str], csv文件路径列表
        """
        self.dirs = output_dirs
        self.output_root = output_root
        self.args = args
        self.hour = hour

    def drawLayer(self, key):
        """
        @param key: which ratio to draw
        """
        if key == "black_p":
            title = "黑样本精准率"
        elif key == "black_r":
            title = "黑样本召回率"
        elif key == "macro_f1":
            title = "macro_f1"
        elif key == "micro_f1":
            title = "micro_f1"
        else:
            raise ValueError("Wrong key...")
        plt.figure()
        layers = []
        values = []
        day = None
        for d in self.dirs:
            day = d.split("_")[1]
            layers.append("layer = " + d.split("_")[3])
            df = pd.read_csv(d)
            values.append(df[key][self.hour])
        print("values: {}".format(values))
        print("layers: {}".format(layers))
        plt.bar(range(len(layers)), values, color="forestgreen", align="center", tick_label=layers, width=0.2)
        plt.title("19-09-{}上第{}小时具有相同维度不同层模型的{}对比".format(day, self.hour, title))
        ax = plt.gca()
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.grid(True, axis="y", linewidth=1, linestyle="-", color="darkslategray")
        plt.savefig(os.path.join(self.output_root, "imgs/bar/day_{}_hour_{}_layers_{}_{}.png") \
                    .format(day, self.hour, len(layers), key))
        plt.close()

    def drawDim(self, key):
        """
        @param ratio: which ratio to draw
        """
        title = None
        if key == "black_p":
            title = "黑样本精准率"
        elif key == "black_r":
            title = "黑样本召回率"
        elif key == "macro_f1":
            title = "macro_f1"
        elif key == "micro_f1":
            title = "micro_f1"
        else:
            raise ValueError("Wrong key...")
        plt.figure()
        dims = []
        values = []
        for d in self.dirs:
            day = d.split("_")[1]
            dims.append("dim = " + d.split("_")[5].split(".")[0])
            df = pd.read_csv(d)
            values.append(df[key][self.hour])
        print("values: {}".format(values))
        print("dims: {}".format(dims))
        plt.bar(range(len(dims)), values, align="center", color="forestgreen", tick_label=dims, width=0.3)
        plt.title("19-09-{}第{}个小时具有相同层不同维度模型的{}对比".format(day, self.hour, title))
        ax = plt.gca()
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.grid(True, axis="y", linewidth=1, linestyle="-", color="darkslategray")
        plt.savefig(os.path.join(self.output_root, "imgs/bar/day_{}_hour_{}_dims_{}_{}.png") \
                    .format(day, self.hour, len(dims), key))
        plt.close()


class AUC():
    def __init__(self, img_dir, data_dirs, args):
        self.img_dir = img_dir
        self.data_dirs = data_dirs
        self.args = args

        self.data = []
        for data_dir in self.data_dirs:
            self.data.append(pd.read_csv(data_dir))

    def drawAUC(self, key="layer"):
        """
        同一个小时不同层次或者不同维度的AUC曲线
        """
        keys = []
        day = self.data_dirs[0].split("_")[1]
        hour = self.data_dirs[0].split("_")[4]
        for i, data in enumerate(self.data):
            label = data["label"]
            prob = []
            for i in range(label.shape[0]):
                if label[i] == 0:
                    prob.append(data["neg_prob"][i])
                else:
                    prob.append(data["pos_prob"][i])
            prob = np.array(prob)
            precision, recall, threshold = roc_curve(label, prob)
            plt.figure()
            plt.plot(precision, recall, linewidth=1)
            if key == "layer":
                keys.append("layer {}: {:.4f}".format(self.data_dirs[i].split("_")[5], roc_auc_score(label, prob)))
            elif key == "dim":
                keys.append("dim {}: {:.4f}".format(self.data_dirs[i].split("_")[7], roc_auc_score(label, prob)))
            else:
                raise ValueError("Wrong key, please check.")
        if key == "layer":
            plt.title("不同层数同一维度模型的AUC对比(19-09-{}第{}小时)" \
                      .format(day, hour))
        elif key == "dim":
            plt.title("不同维度同一层数模型的AUC对比(19-09-{}第{}小时)" \
                      .format(day, hour))
        else:
            raise ValueError("Wrong key, please check.")
        plt.legend(loc="best", labels=keys, frameon=True, framealpha=0.25, title="AUC")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        plt.xlabel("假阳性率")
        plt.ylabel("真阳性率")
        if key == "layer":
            plt.savefig(os.path.join(self.img_dir, "imgs/curves/test_{}_hour_{}_layers_{}_auc.png" \
                                     .format(day, hour, len(self.data_dirs))))
        elif key == "dim":
            plt.savefig(os.path.join(self.img_dir, "imgs/curves/test_{}_hour_{}_dims_{}_auc.png" \
                                     .format(day, hour, len(self.data_dirs))))
        plt.close()


if __name__ == "__main__":
    args = parameter_parser()
    layers = [2, 3, 5, 7]
    dims = [8, 16, 32]
    hours = [4, 8, 15, 23]
    output_root = "/home/zhangpin/project/xiaohongshu/output"
    #     for layer in layers:
    #         for dim in dims:
    #             for hour in hours:
    #                 ratio = Ratio(output_root, layer, dim, hour, args)
    #                 pRecall = PRecall(output_root, layer, dim, hour, args)
    #                 print("start to draw PR curves of hour:{:2d} dim:{:2d} layer:{:2d}...".format(hour, dim, layer))
    #                 pRecall.drawPR()
    #                 print("start to draw ROC curves of hour:{:2d} dim:{:2d} layer:{:2d}...".format(hour, dim, layer))
    #                 pRecall.drawROC()
    #                 keys = ["loss", "acc", "white", "black", "precision_recall", "macro_f1", "micro_f1"]
    #                 for key in keys:
    #                     print("start to draw {} curves of hour:{:2d} dim:{:2d} layer:{:2d}".format(key, hour, dim, layer))
    #                     ratio.draw_ratio(key)

    # # 对比同一维度不同层, 确定维度为32
    # layer_output_dirs = [os.path.join(output_root, "logs/test_02_layer_2_dim_32.csv"),
    #                      os.path.join(output_root, "logs/test_02_layer_3_dim_32.csv"),
    #                      os.path.join(output_root, "logs/test_02_layer_5_dim_32.csv"),
    #                      os.path.join(output_root, "logs/test_02_layer_7_dim_32.csv")]
    #
    # # 对比不同维度同层, 固定层数为7
    # dim_output_dirs = [os.path.join(output_root, "logs/test_02_layer_7_dim_8.csv"),
    #                    os.path.join(output_root, "logs/test_02_layer_7_dim_16.csv"),
    #                    os.path.join(output_root, "logs/test_02_layer_7_dim_32.csv")]
    #
    # # 对比第4个小时不同模型的效果
    # for hour in hours:
    #     layer_ = LayerDim(output_root, layer_output_dirs, hour, args)
    #     dim_ = LayerDim(output_root, dim_output_dirs, hour, args)
    #     keys = ["black_p", "black_r", "macro_f1", "micro_f1"]
    #     for key in keys:
    #         print("start to draw {} curves of different layers...".format(key))
    #         layer_.drawLayer(key)
    #         print("start to draw {} curves of different dims...".format(key))
    #         dim_.drawDim(key)

    for date in ["02", "05"]:
        layer_path = []
        dim_path = []
        for hour in hours:
            layer_path.append(os.path.join(output_root, "curves/test_{}_hour_{}_layer_2_dim_32_pr.csv" \
                                           .format(date, hour)))
            layer_path.append(os.path.join(output_root, "curves/test_{}_hour_{}_layer_3_dim_32_pr.csv" \
                                           .format(date, hour)))
            layer_path.append(os.path.join(output_root, "curves/test_{}_hour_{}_layer_5_dim_32_pr.csv" \
                                           .format(date, hour)))
            layer_path.append(os.path.join(output_root, "curves/test_{}_hour_{}_layer_7_dim_32_pr.csv" \
                                           .format(date, hour)))

            dim_path.append(os.path.join(output_root, "curves/test_{}_hour_{}_layer_7_dim_8_pr.csv" \
                                         .format(date, hour)))
            dim_path.append(os.path.join(output_root, "curves/test_{}_hour_{}_layer_7_dim_16_pr.csv" \
                                         .format(date, hour)))
            dim_path.append(os.path.join(output_root, "curves/test_{}_hour_{}_layer_7_dim_32_pr.csv" \
                                         .format(date, hour)))

        layer_auc = AUC(output_root, layer_path, args)
        layer_auc.drawAUC("layer")

        dim_auc = AUC(output_root, dim_path, args)
        dim_auc.drawAUC("dim")
