# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 4:45 下午
# @Author  : zhangpin
# @Email   : zhangpin@geetest.com
# @File    : node_input_layer
# @Software: PyCharm
import pandas as pd
import torch
import torch.nn as nn


# TODO : GPU or CPU
class NodeInputLayer(nn.Module):
    def __init__(self, embeded_cols=None,
                 embeded_dims=None,
                 category_nums=None,
                 dense_columns=None,
                 linear_map=True,
                 hidden_dim=-1,
                 shared_module=None,
                 context=torch.device("cuda: 1")):
        super(NodeInputLayer, self).__init__()
        output_dim = 0
        module_dict = nn.ModuleDict()
        self.context = context
        self.embedding_cols = []
        self.embedding_dims = []
        self.category_nums = []
        # column that need to be embedded
        if embeded_cols:
            if isinstance(embeded_dims, int):  # all columns embedded to the same dimension
                embeded_dims = [embeded_dims] * len(embeded_cols)
            assert len(embeded_cols) == len(category_nums), "must set category of each column to be embedded. "
            for col, dim, cate_num in zip(embeded_cols, embeded_dims, category_nums):
                if col in module_dict:
                    raise RuntimeError("column {} has already been in module_dict".format(col))
                self.embedding_cols.append(col)
                self.embedding_dims.append(dim)
                self.category_nums.append(cate_num)
                output_dim += dim
                # Embedding layer
                module_dict[col] = nn.Embedding(cate_num, dim)
                # print("embedding {} finished, dimension: {}".format(col, dim))
        if shared_module:
            for k, m in shared_module:
                if k in module_dict:
                    raise ValueError("key: {} has already been in module_dict".format(k))
                if isinstance(m, nn.Embedding):
                    self.embedding_cols.append(k)
                    self.embedding_dims.append(m.embedding_dim)
                    self.category_nums.append(m.num_embeddings)
                    output_dim += m.embedding_dim
            module_dict.update(shared_module)

        if dense_columns:
            output_dim += len(dense_columns)
        self.dense_columns = dense_columns
        self.output_dim = output_dim
        self.module_dict = module_dict
        # TODO: linear layer 线性变换用于升维或者降维
        self.linear_map = linear_map
        if self.linear_map:
            assert hidden_dim > 0
        self.hidden_dim = hidden_dim
        if self.linear_map:
            self.linear = nn.Linear(self.output_dim, self.hidden_dim)

    def forward(self, df: pd.DataFrame):
        embeddings = []
        for col in self.embedding_cols:
            sparse_data = torch.from_numpy(df[col].values).long().to(self.context)
            embed_out = self.module_dict[col](sparse_data)
            embeddings.append(embed_out)

        if self.dense_columns:
            for col in self.embedding_cols:
                dense_data = torch.from_numpy(df[col].values.astype("float32")).view((-1, 1)).to(self.context)
                embeddings.append(dense_data)

        embeddings = torch.cat(embeddings, dim=1)

        if self.linear_map:
            transformed_out = self.linear(embeddings)
            return transformed_out
        return embeddings
