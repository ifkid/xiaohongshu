# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 5:12 下午
# @Author  : zhangpin
# @Email   : zhangpin@geetest.com
# @File    : input_layer
# @Software: PyCharm

import numpy as np
import torch
import torch.nn as nn

from .node_input_layer import NodeInputLayer


class RedBookInputLayer(nn.Module):
    def __init__(self, device_cfg,
                 event_cfg,
                 context,
                 hidden_dim=8):
        super(RedBookInputLayer, self).__init__()
        # node_type is shared by all nodes
        # node type( dimension 5) embedding to dimension 8
        node_type_module = nn.Embedding(5, embedding_dim=hidden_dim)
        share_module = [("encode_node_type", node_type_module)]

        print("start to set up node device...")
        device_embed_cols, device_embed_dims, device_cate_nums = list(zip(*device_cfg))
        node_device = NodeInputLayer(list(device_embed_cols), list(device_embed_dims), list(device_cate_nums),
                                     shared_module=share_module, hidden_dim=hidden_dim, context=context)

        print("start to set up node ip...")
        node_ip = NodeInputLayer(shared_module=share_module, hidden_dim=hidden_dim, context=context)

        print("start to set up node user...")
        node_user = NodeInputLayer(dense_columns=["has_behavior", "is_followed"],
                                   hidden_dim=hidden_dim, context=context)

        print("start to set up node note...")
        node_note = NodeInputLayer(shared_module=share_module, hidden_dim=hidden_dim, context=context)

        print("start to set up node event...")
        event_embed_cols, event_embed_dims, event_cate_nums = list(zip(*event_cfg))
        node_event = NodeInputLayer(list(event_embed_cols), list(event_embed_dims), list(event_cate_nums),
                                    shared_module=share_module, hidden_dim=hidden_dim, context=context)

        self.context = context
        self.node_device_layer = node_device
        self.node_ip_layer = node_ip
        self.node_user_layer = node_user
        self.node_note_layer = node_note
        self.node_event_layer = node_event

    def forward(self, device_df, ip_df, note_df, user_df, event_df):
        device_id = device_df["device_id_node_id"].values
        ip_id = ip_df["user_ipaddress_node_id"].values
        note_id = note_df["note_node_id"].values
        user_id = user_df["user_node_id"].values
        event_id = event_df["event_id_node_id"].values
        node_id = np.concatenate((device_id, ip_id, user_id, note_id, event_id))
        node_id = torch.from_numpy(node_id).long().to(self.context)

        x_device = self.node_device_layer(device_df)
        x_ip = self.node_ip_layer(ip_df)
        x_user = self.node_user_layer(user_df)
        x_note = self.node_note_layer(note_df)
        x_event = self.node_event_layer(event_df)
        X = torch.cat((x_device, x_ip, x_user, x_note, x_event), dim=0)

        output = torch.zeros_like(X)
        output[node_id] = X
        return output
