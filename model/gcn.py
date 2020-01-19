import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from data import embedding_table as et
from data.redbook_input_layer import RedBookInputLayer


class GCN(torch.nn.Module):
    def __init__(self, args, context, layer, dim):
        super(GCN, self).__init__()
        self.device_cfg = self.configDevice()
        self.event_cfg = self.configEvent()
        self.layer = layer
        self.gcn_dim = dim

        self.input_layer = RedBookInputLayer(device_cfg=self.device_cfg, event_cfg=self.event_cfg, context=context)
        if self.layer == 2:
            self.conv1 = GCNConv(args.input_dim, self.gcn_dim, cached=False, improved=True)
            self.conv2 = GCNConv(self.gcn_dim, args.output_dim, cached=False, improved=True)

        # because of weak GPU memory, 7 layer with dim 32 is different
        elif self.layer == 7 and self.gcn_dim == 32:
            self.conv1 = GCNConv(args.input_dim, int(args.input_dim*2), cached=False, improved=True)
            self.conv2 = GCNConv(int(2*args.input_dim), self.gcn_dim, cached=False, improved=True)
            self.conv3 = GCNConv(self.gcn_dim, self.gcn_dim, cached=False, improved=True)
            self.conv4 = GCNConv(self.gcn_dim, int(self.gcn_dim/2), cached=False, improved=True)
            self.conv5 = GCNConv(int(self.gcn_dim/2), int(self.gcn_dim/2), cached=False, improved=True)
            self.conv6 = GCNConv(int(self.gcn_dim/2), args.input_dim, cached=False, improved=True)
            self.conv7 = GCNConv(args.input_dim, args.output_dim, cached=False, improved=True)
        else:
            self.conv1 = GCNConv(args.input_dim, self.gcn_dim, cached=False, improved=True)
            for l in range(2, self.layer):
                setattr(self, "conv{}".format(l), GCNConv(self.gcn_dim, self.gcn_dim, cached=False, improved=True))
            setattr(self, "conv{}".format(self.layer), GCNConv(self.gcn_dim, args.output_dim, cached=False, improved=True))

    def forward(self, inputs_df, edge_index):
        x = self.input_layer(*inputs_df)
        x = F.relu(self.conv1(x, edge_index))
        if self.layer == 2:
            feature_map = x
            x = self.conv2(x, edge_index)
        else:
            for l in range(2, self.layer):
                x = F.relu(getattr(self, "conv{}".format(l))(x, edge_index))
            feature_map = x
            x = getattr(self, "conv{}".format(self.layer))(x, edge_index)
        return x, feature_map

    def configDevice(self):
        device_cfg = [["encode_andr_channel", et["encode_andr_channel"], 8],
                      ["encode_app_id", et["encode_app_id"], 8],
                      ["encode_device_model", et["encode_device_model"], 16],
                      ["encode_os_version", et["encode_os_version"], 10],
                      ["encode_dvce_manufacturer", et["encode_dvce_manufacturer"], 10]]
        return device_cfg

    def configEvent(self):
        event_cfg = [["encode_event_sub_type", et["encode_event_sub_type"], 8],
                     ["collector_hour", 24, 8],
                     ["collector_minute", 60, 8]]
        return event_cfg
