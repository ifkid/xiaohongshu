from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from data.redbook_input_layer import RedBookInputLayer

from data import embedding_table as et


class GCN(torch.nn.Module):
    def __init__(self, args, context=torch.device("cuda:1")):
        super(GCN, self).__init__()
        self.device_cfg = self.configDevice()
        self.event_cfg = self.configEvent()

        self.input_layer = RedBookInputLayer(device_config=self.device_cfg, event_config=self.event_cfg, context=context)
        self.conv1 = GCNConv(args.input_dim, args.gcn_dim, cached=False, improved=True)
        self.conv2 = GCNConv(args.gcn_dim, args.gcn_dim, cached=False, improved=True)
        self.conv3 = GCNConv(args.gcn_dim, args.gcn_dim, cached=False, improved=True)
        self.conv4 = GCNConv(args.gcn_dim, args.gcn_dim, cached=False, improved=True)
        self.conv5 = GCNConv(args.gcn_dim, args.output_dim, cached=False, improved=True)

    #         self.conv6 = GCNConv(args.gcn_dim, args.input_dim, cached=False, improved=True)
    #         self.conv7 = GCNConv(args.input_dim, args.output_dim, cached=False, improved=True)

    def forward(self, inputs_df, edge_index):
        x = self.input_layer(*inputs_df)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = F.relu(self.conv3(x, edge_index))
        #         x = F.dropout(x, training=self.training)
        x = F.relu(self.conv4(x, edge_index))
        #         x = F.relu(self.conv5(x, edge_index))
        #         x = F.relu(self.conv6(x, edge_index))
        #         x = self.conv7(x, edge_index)
        x = self.conv5(x, edge_index)
        return x

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