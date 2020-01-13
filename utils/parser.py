# -*- coding: utf-8 -*-
# @Time    : 2019/12/10 2:16 下午
# @Author  : zhangpin@Geetest
# @FileName: parser.py
import argparse
from texttable import Texttable


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format
    @param args: Parameters used in this model
    @return:
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " "), args[k]] for k in keys])
    print(t.draw())


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run.")
    parser.add_argument("--epochs", type=int, default=200, help="epochs of training, default is 20.")
    parser.add_argument("--early_stop", type=int, default=200, help="epoch of early when train, default is 20.")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate when train, default is 0.01.")
    parser.add_argument("--input_dim", type=int, default=8, help="input dimension of model, default is 8.")
    parser.add_argument("--gcn_dim", type=int, default=8, help="hidden gcn dimension of model, default is 16.")
    parser.add_argument("--output_dim", type=int, default=2, help="output dimension, must be 2.")
    parser.add_argument("--dpi", type=int, default=800, help="dpi of saving figure, default is 800")
    parser.add_argument("--layer", type=int, default=5, help="layer of gcn, default is 7")

    return parser.parse_args()
