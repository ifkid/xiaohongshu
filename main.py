# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 6:06 下午
# @Author  : zhangpin
# @Email   : zhangpin@geetest.com
# @File    : main
# @Software: PyCharm

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data.data_loader import DataSet
from model.gcn import GCN
from utils.parser import parameter_parser, tab_printer
from utils.tool import evaluate, save_model, norm


def train(model_save_path, root_dir, day, train_log_dir, args, layer, dim):
    # print params
    tab_printer(args)
    # load train dataset
    dataset = DataSet(os.path.join(root_dir, day))  # TODO: 更改训练数据

    # create model on GPU:1
    running_context = torch.device("cuda:0")
    model = GCN(args, running_context, layer, dim).to(running_context)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    # train logs
    train_log_list = np.zeros((args.epochs, 24), dtype=np.float)

    print("\n" + "+" * 5 + " Train on day {} layer {} dim {} ".format(day, layer, dim) + "+" * 5)
    best_loss = float("inf")
    last_loss_decrease_epoch = 0
    stop_flag = False
    best_model = None
    train_start_time = time.perf_counter()
    for epoch in range(args.epochs):
        start_time = time.perf_counter()
        total_loss = 0.
        print("\n" + "+" * 10 + " epoch {:3d} ".format(epoch) + "+" * 10)
        for i in range(len(dataset)):  # 24 hours
            data = dataset[i]
            edge_index = data.edge_index.to(running_context)
            mask = data.mask.to(running_context)
            logits = model(data.inputs, edge_index)
            label = data.label.to(running_context, non_blocking=True)
            pos_cnt = torch.sum(label == 1)
            neg_cnt = torch.sum(label == 0)
            weight = torch.tensor([pos_cnt.float(), neg_cnt.float()]) / (neg_cnt + pos_cnt)
            loss_fn = nn.CrossEntropyLoss(weight=weight).to(running_context)
            loss = loss_fn(logits[mask], label)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #             acc, tn, fp, fn, tp, white_p, white_r, white_incorrect, black_p, black_r, black_incorrect, macro_f1, micro_f1 \
            #                 = evaluate(logits[mask].max(1)[1], label)

            train_log_list[epoch][i] = float(loss.item())
            print("hour: {:2d}, loss: {:.4f}".format(i, loss.item()))

        if total_loss < best_loss:
            best_loss = total_loss
            last_loss_decrease_epoch = epoch
            best_model = model.state_dict()
        else:
            not_loss_decrease_epochs = epoch - last_loss_decrease_epoch
            if not_loss_decrease_epochs >= args.early_stop:
                stop_flag = True
            else:
                pass
        if stop_flag:
            print("early stop...")
            save_model(best_model, model_save_path)
        print("\nepoch: {:3d}, total_loss: {:.4f}, best_loss: {:.4f} time: {:.4f}" \
              .format(epoch + 1, total_loss, best_loss, time.perf_counter() - start_time))
    print("\ntotal train time: {}".format(time.perf_counter() - train_start_time))
    # save model when not early stop
    if not stop_flag:
        save_model(best_model, model_save_path)

    # save train logs to csv file
    print("Start to save train log of {}.".format(day))
    train_log_cols = ["hour_{}".format(hour) for hour in range(24)]
    train_log_df = pd.DataFrame(train_log_list, columns=train_log_cols)
    train_log_df.to_csv(train_log_dir, float_format="%.4f", index=None, columns=train_log_cols)
    print("Save train log of {} layer {} dim {} successfully.".format(day, layer, dim))
    torch.cuda.empty_cache()


def test(model_save_path, root_dir, p_r_root, day, test_log_dir, args, layer, dim):
    # load test dataset
    dataset = DataSet(os.path.join(root_dir, day))  # TODO: 测试不同的数据
    # load model
    running_context = torch.device("cuda:1")
    model = GCN(args, running_context, layer, dim).to(running_context)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    # test log
    test_log_list = [[] for _ in range(24)]
    test_count = 0
    print("\n" + "+" * 20 + " Test on day {} layer {} dim {} ".format(day, layer, dim) + "+" * 20)
    for hour in range(0, 24):
        data = dataset[hour]
        edge_index = data.edge_index.to(running_context)
        mask = data.mask.to(running_context)
        logit = model(data.inputs, edge_index)
        label = data.label.to(running_context, non_blocking=True)
        acc, tn, fp, fn, tp, white_p, white_r, white_incorrect, black_p, black_r, black_incorrect, macro_f1, micro_f1 = evaluate(logit[mask].max(1)[1], label)

        test_log_list[test_count].append(float(acc))
        test_log_list[test_count].append(float(white_p))
        test_log_list[test_count].append(float(white_r))
        test_log_list[test_count].append(float(white_incorrect))
        test_log_list[test_count].append(float(black_p))
        test_log_list[test_count].append(float(black_r))
        test_log_list[test_count].append(float(black_incorrect))
        test_log_list[test_count].append(float(macro_f1))
        test_log_list[test_count].append(float(micro_f1))
        test_count += 1
        test_log = "hour:{:3d}, acc: {:.4f}" \
                   "TN={:6d}, FP={:6d}, FN={:6d}, TP={:6d}, " \
                   "white_p={:.4f}, white_r={:.4f}, " \
                   "black_p={:.4f}, black_r={:.4f}, " \
                   "macro_f1: {:.4f}, micro_f1: {:.4f}"
        print(test_log.format(hour, acc, tn, fp, fn, tp, white_p, white_r, black_p, black_r, macro_f1, micro_f1))

        logit = np.array(norm(torch.sigmoid(logit[mask]).cpu().detach().numpy()))
        label = np.array(label.cpu().detach().numpy()).reshape(-1, 1)
        p_r = np.concatenate((logit, label), axis=1)
        p_r_cols = ["neg_pro", "pos_pro", "label"]
        p_r_df = pd.DataFrame(np.array(p_r), columns=p_r_cols)
        p_r_dir = os.path.join(p_r_root, "test_{}_hour_{}_layer_{}_dim_{}_pr.csv" \
                               .format(day.split("-")[-1], hour, layer, dim))

        p_r_df.to_csv(p_r_dir, index=None, columns=p_r_cols)
    # save test logs to csv file
    print("Start to save test log of {}.".format(day))
    test_log_cols = ["acc", "white_p", "white_r", "white_incorrect", "black_p", "black_r", "black_incorrect", "macro_f1", "micro_f1"]
    test_log_df = pd.DataFrame(np.array(test_log_list), columns=test_log_cols)
    test_log_df.to_csv(test_log_dir, float_format="%.4f", index=None, columns=test_log_cols)
    print("Save test log of day {} layer {} dim {} successfully.".format(day, layer, dim))
    torch.cuda.empty_cache()


if __name__ == "__main__":

    # init params
    args = parameter_parser()
    data_root = "/home/zhangpin/data/redbook"
    train_day = "19-09-01"
    test_day = ["19-09-02", "19-09-05"]
    p_r_root = "/home/zhangpin/project/xhs/output/precision_recall"
    layers = [2, 3, 4, 5, 6, 7]
    dims = [4, 8, 16, 24, 32]

    # train and test
    for layer in layers:
        for dim in dims:
            assert layer >= 2, "layer must more than 2"
            assert dim >= 2, "dim must more than 2"
            args.layer = layer
            args.dim = dim
            train_log_dir = "/home/zhangpin/project/xhs/output/logs/train_{}_layer_{}_dim_{}.csv" \
                .format(train_day.split("-")[-1], layer, dim)
            model_save_path = "/home/zhangpin/project/xhs/checkpoint/gcn_day_{}_layer_{}_dim_{}.pt" \
                .format(train_day.split("-")[-1], layer, dim)
            # train on 19-09-01
            train(model_save_path, data_root, train_day, train_log_dir, args, layer, dim)
            # test on 19-09-02/19-09-05
            for day in test_day:
                test_log_dir = "/home/zhangpin/project/xhs/output/logs/test_{}_layer_{}_dim_{}.csv" \
                    .format(day.split("-")[-1], layer, dim)
                test(model_save_path, data_root, p_r_root, day, test_log_dir, args, layer, dim)
