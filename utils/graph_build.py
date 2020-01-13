# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 9:20 上午
# @Author  : zhangpin
# @Email   : zhangpin@geetest.com
# @File    : build_graph
# @Software: PyCharm
import glob
import os
import pickle as pkl
from collections import namedtuple

import pandas as pd


def assign_node_id(df: pd.DataFrame, key, start_index):
    """
    assign id to node
    @param df:
    @param key:
    @param start_index:
    @return:
    """
    num_rows = len(df)
    end_index = start_index + num_rows
    indices = range(start_index, end_index)
    df["{}_node_id".format(key)] = indices
    return df, end_index


def read_csv(file_dir, dates=None):
    """
    read csv file
    @param file_dir:
    @param dates:
    @return:
    """
    file_name = glob.glob(os.path.join(file_dir, "*.csv"))
    df = pd.read_csv(file_name[0], parse_dates=dates, header=0)
    return df


def read_user(user_path):
    """
    read node user
    @param user_path:
    @return:
    """
    print("Start to read user...")
    user_df = read_csv(user_path)
    # 对一些字段进行填充
    user_df["has_behavior"] = user_df.has_behavior.fillna(0).astype("float32")
    user_df["isFollow"] = user_df.isFollw.fillna(0).astype("float32")
    user_df["user_app_first_time"] = user_df.user_app_first_time.fillna("1970-01-01 00:00:00.000")
    user_df["user_create_time"] = user_df.user_create_time.fillna("1970-01-01 00:00:00.000")
    return user_df


def read_note(note_path):
    """
    read node note
    @param note_path:
    @return:
    """
    print("Start to read note...")
    note_df = read_csv(note_path)
    mask = note_df["discovery_id_create_time"].notnull()
    note_df.loc[mask, "discovery_id_create_year_month"] = \
        pd.to_datetime(note_df.loc[mask, "discovery_id_create_time"]).map(lambda x: x.strftime("%y-%m"))
    note_df["discovery_id_create_year_month"].fillna("", inplace=True)
    return note_df


def read_ip(ip_path):
    """
    read node ip
    @param ip_path:
    @return:
    """
    print("Start to red ip...")
    ip_df = read_csv(ip_path)
    return ip_df


def read_device(device_path):
    """
    read node device
    @param device_path:
    @return:
    """
    print("Start to read device...")
    device_df = read_csv(device_path)
    # 填充缺失字段
    device_df["connection_type"] = device_df.connection_type.fillna("other")
    device_df.loc[device_df.dvce_manufacturer == "Apple Inc.", "os_version"] = \
        device_df.loc[device_df.dvce_manufacturer == "Apple Inc.", "os_version"].map(lambda x: "Apple_" + x)

    device_df.loc[device_df.dvce_manufacturer != "Apple Inc.", "os_version"] = \
        device_df.loc[device_df.dvce_manufacturer != "Apple Inc.", "os_version"].fillna("")

    device_df.loc[device_df.dvce_manufacturer != "Apple Inc.", "os_version"] = \
        device_df.loc[device_df.dvce_manufacturer != "Apple Inc.", "os_version"].map(lambda x: "Android_" + x)
    return device_df


def read_hour_event(event_paths):
    """
    read node event by hour
    @param event_path:
    @return:
    """
    events = []
    for path in event_paths:
        hour_event = read_csv(path, dates=["collector_tstamp", "dvce_tstamp"])
        hour_event["collector_hour"] == hour_event["collector_tstamp"].map(lambda x: x.hour)
        hour_event["collector_minute"] == hour_event["collector_tstamp"].map(lambda x: x.minute)
        events.append(hour_event)
    event_df = pd.concat(events)
    event_df = event_df.loc([event_df.user_token.notnull()])
    return event_df


def read_node_df(data_root):
    user_path = data_root + "node_user.csv/"
    note_path = data_root + "node_note.csv/"
    device_path = data_root + "node_device.csv/"
    ip_path = data_root + "node_ip.csv/"

    user = read_user(user_path)
    note = read_note(note_path)
    device = read_device(device_path)
    ip = read_ip(ip_path)

    return user, note, device, ip


def load_pickle(Data):
    for h in range(24):
        p = "{}_data.pkl".format(h)
        with open(p, "rb") as f:
            data = pkl.load(f)
            print("hour: {:2d}, nodes: {:8d}, edge_index: {}".format(h, data.num_nodes, data.edge_index))


class GraphBuilder():
    def __init__(self, data_root, user, note, device, ip, hour):
        """
        class of build graph
        @param data_root:
        @param user: all user node
        @param note: all note node
        @param device: all device node
        @param ip: all ip node
        @param hour:
        """
        save_root = os.path.join(data_root, "pickles")
        data_root = os.path.join(data_root, "event_by_user.csv/")
        os.makedirs(save_root, exist_ok=True)
        if isinstance(hour, int):
            hour_event_path = [data_root + "hour_{}/".format(hour)]
        elif isinstance(hour, list):
            hour_event_path = [data_root + "hour_{}/".format(h) for h in hour]
        else:
            raise ValueError("Unknown format of hour.")
        print(" Start to process {}...".format(hour_event_path))
        self.hour = hour
        self.save_root = save_root

        # 读取每个小时的事件节点，并对其进行编号
        hour_event = read_hour_event(hour_event_path)
        self.hour_event, self.current_index = assign_node_id(hour_event, "event_id", 0)

        self.all_user = user
        self.all_note = note
        self.all_device = device
        self.all_ip = ip

        self.part_user = None
        self.part_note = None
        self.part_device = None
        self.part_ip = None
        self.all_edges = None
        self.num_nodes = 0

        edge_config = [("event_id_node_id", "user_ipaddress_node_id"),
                       ("event_id_node_id", "device_id_node_id"),
                       ("event_id_node_id", "user_node_id"),
                       ("event_id_node_id", "destination_node_id")]

        self.split_nodes()
        self.join_edge()
        self.get_edges(edge_config)
        self.pickle_to_file()

    def split_nodes(self):
        self.part_device = self._get_part_node("device_id")
        self.part_ip = self._get_part_node("user_ipaddress")
        self.part_user = self._get_part_node("user")
        self.part_note = self._get_part_node("note")

        self.num_nodes = self.current_index

    def _get_part_node(self, key):
        """
        设置每个小时内数据包含的节点，device/ip/user/note，设置统一编号
        @param key:
        @return:
        """
        if key in ["user_ipaddress", "device_id"]:
            mask = self.hour_event[key].notnull()
            unique_node = self.hour_event.loc[mask, key].drop_duplicates().to_frame()
            unique_node, current_index = assign_node_id(unique_node, key, self.current_index)
            unique_node_with_property = unique_node.merge(self.__dict__[key], on=key, how="left")
            self.current_index = current_index
            return unique_node_with_property
        elif key == "user":
            mask_source = self.hour_event.user_token.notnull()
            source_user = self.hour_event.loc[:, "user_token"].drop_duplicates().to_frame()
            mask_dest = (self.hour_event.se_property_type == "user") & self.hour_event.dst_node.notnull()
            dest_user = self.hour_event.loc[mask_dest, "dst_node"].drop_duplicates().to_frame() \
                .rename(columns={"dst_node": "user_token"})
            users = pd.concat([source_user, dest_user]).drop_duplicates()
            user, current_index = assign_node_id(users, key, self.current_index)
            user_with_property = users.merge(self.all_user, on="user_token", how="left")
            self.current_index = current_index
            return user_with_property
        elif key == "note":
            mask = (self.hour_event.se_property_type == "note") & self.hour_event.dst_node.notnull()
            notes = self.hour_event.loc[mask, "dst_node"].drop_duplicates().to_frame() \
                .rename(columns={"dst_node": "discovery_id"})
            notes, current_index = assign_node_id(notes, key, self.current_index)
            note_with_property = notes.merge(self.all_note, on="discovery_id", how="left")
            self.current_index = current_index
            return note_with_property
        else:
            raise ValueError("Unknown key.")

    def join_edge(self):
        for key, node_data in zip(["user_ipaddress", "device_id"], [self.part_ip, self.part_device]):
            print(" merge node: {}".format(key))
            cols = [key, "{}_node_id".format(key)]
            node_data_id_part = node_data[cols]
            self.hour_event = self.hour_event.merge(node_data_id_part, on=key, how="left")

        # merge user
        print(" merge node user...")
        # 1. 实施行为的user
        cols = ["user_token", "user_node_id"]
        node_user_id_part = self.part_user[cols]
        self.hour_event = self.hour_event.merge(node_user_id_part, on="user_token", how="left")
        # 2. 被关注的用户关联
        mask_followed_user = (self.hour_event.se_property_type == "user") & (self.hour_event.dst_node.notnull())
        dest_user_edges = self.hour_event.loc[mask_followed_user, ["event_id_node_id", "dst_node"]] \
            .rename(columns={"dst_node": "user_token"}) \
            .merge(node_user_id_part, on="user_token", how="left")
        self.hour_event["destination_node_id"] = -1
        self.hour_event.loc[mask_followed_user, "destination_node_id"] = dest_user_edges.user_node_id.values

        # merge note
        print(" merge node note...")
        cols = ["discovery_id", "note_node_id"]
        node_note_id_part = self.part_note[cols]
        mask_note = (self.hour_event.se_property_type == "note") & self.hour_event.dst_node.notnull()
        notes_edges = self.hour_event.loc[mask_note, ["event_id_node_id", "dst_node"]] \
            .rename(columns={'dst_node': 'discovery_id'}) \
            .merge(node_note_id_part, on="discovery_id", how="left")  # event_id_node_id, discovery_id, note_node_id
        self.hour_event.loc[mask_note, "destination_node_id"] = notes_edges.note_node_id.values
        self.hour_event["destination_node_id"] = self.hour_event["destination_node_id"].astype("int")

    def pickle_to_file(self, Data):
        data = Data(device=self.part_device,
                    ip=self.part_ip,
                    user=self.part_user,
                    note=self.part_note,
                    event=self.hour_event,
                    edge_index=self.all_edges,
                    num_nodes=self.num_nodes)
        print(" save to files...")
        save_path = os.path.join(self.save_root, "{}_data.pkl".format(self.hour))
        if os.path.exists(save_path):
            os.remove(save_path)
        with open(save_path, "wb") as f:
            pkl.dump(data, f)

    def get_edges(self, edge_config):
        edges = []
        for cols in edge_config:
            mask = self.hour_event[cols[0]].notnull() & self.hour_event[cols[1]].notnull()
            edge = self.hour_event.loc[mask, list(cols)]
            edge.columns = ["src_id", "dst_id"]
            edges.append(edge)
        edges = pd.concat(edges).drop_duplicates()
        self.all_edges = edges


if __name__ == "__main__":
    for day in range(1, 6):  # TODO: 修改 19-09-01 ~ 19-09-05
        date = "19-09-0{}".format(day)
        root_dir = "/home/zhangpin/xiaohongshu/{}".format(date)
        # 具名元组
        Data = namedtuple("Data", ["device", "ip", "user", "note", "event", "edge_index", "num_nodes"])
        node_user, node_note, node_device, node_ip = read_node_df(root_dir)
        print("+" * 30 + " day {} ".format(date) + "+" * 30)
        for hour in range(24):
            GraphBuilder(root_dir, node_user, node_note, node_device, node_ip, hour)
