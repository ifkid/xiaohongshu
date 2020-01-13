# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 3:48 下午
# @Author  : zhangpin
# @Email   : zhangpin@geetest.com
# @File    : data_loader
# @Software: PyCharm

import os
import pickle as pkl
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from . import embedding_table as ET

Data = namedtuple("Data", ["device", "ip", "user", "note", "event",
                           "edge_index", "num_nodes"])


class RedBookData():
    def __init__(self, path):
        data: Data = pkl.load(open(path, "rb"))
        self.data = data
        self.device = self.process_device()
        self.ip = self.process_ip()
        self.note = self.process_note()
        self.user = self.process_user()
        self.event = self.process_event()
        self.inputs = (self.device, self.ip, self.note, self.user, self.event)  # 5 nodes
        self.edge_index = self.build_edge_index()  # adj matrix
        self.label = torch.from_numpy(self.event["isSuspect"].to_numpy())
        mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
        # mask标注event_id的node_id
        mask[torch.from_numpy(self.event["event_id_node_id"].to_numpy())] = True
        self.mask = mask

    def process_device(self):
        device_embed_cols = ['andr_channel', "app_id", "device_model", "os_version", "dvce_manufacturer", "node_type"]
        device = ET.encode_embed_columns(self.data.device, device_embed_cols)
        return device

    def process_ip(self):
        ip = ET.encode_embed_columns(self.data.ip, ["node_type"])
        return ip

    def process_note(self):
        note = ET.encode_embed_columns(self.data.note, ["node_type"])
        return note

    def process_user(self):
        user = ET.encode_embed_columns(self.data.user, ["node_type"])
        return user

    def process_event(self):
        self.data.event["node_type"] = "event"
        event = ET.encode_embed_columns(self.data.event, ["event_sub_type", "node_type"])
        return event

    def build_edge_index(self):
        """
        index of edge
        torch都是双向边
        """
        raw_dege_index = self.data.edge_index
        raw_dege_index = raw_dege_index.loc[raw_dege_index.dst_id.notnull()]  # 过滤出非空
        edge_index = raw_dege_index.to_numpy(dtype="int").T
        edge_index_reverse = edge_index[(1, 0), :]
        all_edges = np.concatenate((edge_index, edge_index_reverse), axis=1)
        return torch.from_numpy(all_edges).long()


class DataSet(Dataset):
    def __init__(self, data_root):
        pkl_files = [os.path.join(data_root, "{}_data.pkl".format(hour)) for hour in range(24)]
        self.keys = [os.path.basename(fn).split(".")[0] for fn in pkl_files]
        self._files_map = dict(zip(self.keys, pkl_files))
        self._dataset_map = {}

    def __getitem__(self, index):
        key = self.keys[index]
        if key not in self._dataset_map:
            data = RedBookData(self._files_map[key])
            self._dataset_map[key] = data
        return self._dataset_map[key]

    def __len__(self):
        return len(self._files_map)


class Dataloader():
    def __init__(self, dataset: DataSet):
        self.dataset = dataset
        for i in range(len(self.dataset)):
            data = dataset[i]
            train_data=Data(x=data.inputs, edge_index=data.edge_index,
                            y=data.label,)