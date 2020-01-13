# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 3:58 下午
# @Author  : zhangpin
# @Email   : zhangpin@geetest.com
# @File    : embedding_table
# @Software: PyCharm
import glob
import os

import pandas as pd


class EmbeddingTable():
    def __init__(self, embedding_dir):
        files = glob.glob(os.path.join(embedding_dir, "*.csv"))
        embedding_cols = [os.path.basename(f).split(".")[0] for f in files]
        self.embedding_cols = embedding_cols
        self.files = files
        self.embedding_dir = embedding_dir
        self._num_embeddings = {}
        self.embedding_table = self.read_embedding_tables()

    def read_embedding_tables(self):
        embedding_table = {}
        for col in self.embedding_cols:
            table = pd.read_csv(os.path.join(self.embedding_dir, col + ".csv"), header=None)
            table.columns = [col]
            table["encode_" + col] = range(1, 1 + len(table))  # 编号
            embedding_table[col] = table
            # 全局
            self._num_embeddings["encode_" + col] = len(table) + 1

        node_type = pd.DataFrame(["ip", "device", "user", "discovery", "event"], columns=["node_type"])
        node_type["encode_node_type"] = range(len(node_type))
        event_sub_type = pd.DataFrame(["like", "follow", "collect", "share", "logon"], columns=["event_sub_type"])
        event_sub_type["encode_event_sub_type"] = range(len(event_sub_type))
        embedding_table["node_type"] = node_type
        embedding_table["event_sub_type"] = event_sub_type
        self._num_embeddings["encode_node_type"] = 5
        self._num_embeddings["encode_event_sub_type"] = 5

        return embedding_table

    def encode_embed_columns(self, df: pd.DataFrame, embed_cols):
        for col in embed_cols:
            add_col = "encode_" + col
            df = df.merge(self.embedding_table[col], on=col, how="left")
            df.loc[:, add_col] = df.loc[:, add_col].fillna(0).astype("int")
        return df

    def __getitem__(self, key):
        return self._num_embeddings[key]
