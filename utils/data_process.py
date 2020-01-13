# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 3:45 下午
# @Author  : zhangpin
# @Email   : zhangpin@geetest.com
# @File    : data_process
# @Software: PyCharm

import os
import urllib.parse as parse
import warnings

import pyspark.sql.functions  as F
import pyspark.sql.types  as T
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

warnings.filterwarnings("ignore")
spark = SparkSession.builder.config(SparkConf()).getOrCreate()


@F.udf(T.StringType())
def parse(encode_s):
    return parse.unquote(encode_s) if encode_s else encode_s


def save2csv(df: DataFrame, save_root, file_name):
    save_file = os.path.join(save_root, file_name)
    df.write.csv(save_file, mode="overwrite", header=None)
    print("{} has been saved in {} successfully.".format(file_name, save_root))


class DataProcess():
    def __init__(self, data_path, save_dir):
        """
        init dataFrame
        @param data_path: path of source data
        @param save_dir: path of save files
        """
        df_init = spark.read.parquet(data_path)
        df_init = self.preprocess(df_init)
        print(df_init.groupBy("isSuspect").count().toPandas())
        self.df_init = df_init
        self.save_dir = save_dir

    def preprocess(self, df: DataFrame):
        preprocess_df = df.filter(~F.isnull("se_property")) \
            .withColumn("se_label", F.lower(F.col("se_label"))) \
            .filter(~F.isnull("se_label")) \
            .withColumn("se_property_type", self.classify("event_sub_type", "se_label")) \
            .withColumn("isSuspect", F.col("isSuspect").cast("int")) \
            .filter(F.col("author_id").isNotNull() | F.col("discovery_id").isNotNull()) \
            .drop_duplicates(["event_id", "user_token", "device_id", "user_ipaddress", "isSuspect"]) \
            .withColumn("hour", F.hour("collector_tstamp"))
        return preprocess_df

    @staticmethod
    @F.udf(T.StringType())
    def classify(event_type, se_label):
        """
        classify user and note
        @param self:
        @param event_type:
        @param se_label:
        @return:
        """
        # when event_type is follow,
        if event_type == "follow":
            return "user"
        # when event_type is share and se_label is user
        elif event_type == "share" and se_label == "user":
            return "user"
        else:
            return "note"

    def process_device(self):
        """
        process node device
        @return:
        """

        @F.udf(T.ArrayType(T.StringType()))
        def fillApple(dvce_manufacturer: str, andr_channel, os_timezone):
            """
            when manufacturer is Apple, fill andr_channel and os_timezone
            is "AppStore" and "Asia/Shanghai"
            @param dvce_manufacturer:
            @param andr_channel:
            @param os_timezone:
            @return:
            """
            if "Apple" in dvce_manufacturer:
                return "AppStore", "Asia/Shanghai"
            else:
                return andr_channel, os_timezone

        device_cols = ["device_id", "andr_channel", "android_id", "app_id",
                       "connection_type", "device_model", "os_version",
                       "dvce_manufacturer", "os_timezone", "ua_parse"]

        device_df = self.df_init.select(*device_cols) \
            .filter(~F.isnull("device_id")) \
            .drop_duplicates(["device_id"]) \
            .withColumn("node_type", F.lit("device")) \
            .withColumn("connection_type", parse("connection_type")) \
            .withColumn("fill_apple", fillApple("dvce_manufacturer", "andr_channel", "os_timezone")) \
            .withColumn("andr_channel", F.col("fill_apple").getItem(0)) \
            .withColumn("os_timezone", F.col("fill_apple").getItem(1)) \
            .drop("fill_apple")

        save2csv(device_df.repartition(1), self.save_dir, "node_device.csv")
        print("node device has been processed successfully.\n")
        return device_df

    def process_ip(self):
        """
        process node ip
        @return:
        """
        ip_cols = ["user_ipaddress"]
        ip_df = self.df_init.select(*ip_cols) \
            .filter(~F.isnull("user_ipaddress")) \
            .drop_duplicates(["user_ipaddress"]) \
            .withColumn("node_type", "ip")
        save2csv(ip_df.repartition(1), self.save_dir, "node_ip.csv")
        print("node device has been processed successfully.\n")
        return ip_df

    def process_user(self, behave_dir, active_dir):
        """
        process node user
        @return:
        """
        source_user = self.df_init.select("user_token", "user_app_first_time", "user_create_time") \
            .filter(~F.isnull("user_token")) \
            .drop_duplicates(["user_token"]) \
            .withColumn("has_behavior", F.lit(1))

        dest_user = self.df_init.filter(F.col("se_property_type") == "user") \
            .select("se_property").drop_duplicates() \
            .select(F.md5("se_property").alias("user_token")) \
            .withColumn("isFollow", F.lit(1))
        user_df = source_user.join(dest_user, on="user_token", how="outer").withColumn("noe_type", F.lit("user"))
        save2csv(user_df.repartition(1), self.save_dir, "node_user.csv")
        print("node user has been processed successfully.\n")
        return user_df

    def process_note(self):
        """
        process node note
        @return:
        """
        note_df = self.df_init.filter(F.col("se_property_type") == "note") \
            .select("se_property", "discovery_id_create_time", "discovery_id_update_time") \
            .drop_duplicates(["se_property"]) \
            .withColumnRename("se_property", "discovery_id") \
            .withColumn("node_type", F.lit("discovery"))
        save2csv(note_df.repartition(1), self.save_dir, "node_discovery.csv")
        print("node device has been processed successfully.\n")
        return note_df

    def process_event(self):
        event_cols = ["event_id", 'collector_tstamp', 'domain_sessionid', 'dvce_tstamp',
                      'event_sub_type', 'se_label', 'se_property_type', 'user_token',
                      'device_id', 'user_ipaddress', 'hour', 'se_property',
                      'isSuspect', 'suspect_reason']
        #  with  user
        event_to_user = self.df_init.filter(F.col("se_property_type") == "user") \
            .select(*event_cols).withColumn("dst_node", F.md5("se_property")) \
            .drop_duplicates(["event_id"])

        # with note
        event_to_note = self.df_init.filter(F.col("se_property_type") == "note") \
            .select(*event_cols).withColumn("dst_node", F.md5("se_property")) \
            .drop_duplicates(["event_id"]).select(event_to_user.columns)

        event_df = event_to_user.unionAll(event_to_note)
        # 分小时存csv
        event_df.repartition(1).write.partitionBy("hour").csv(os.path.join(self.save_dir, "event_by_hour.csv"))
        print("node device has been processed successfully.\n")
        return event_df


if __name__ == "__main__":
    for day in range(1, 6):  # TODO: 修改
        date = "19-09-0{}".format(day)
        data_dir = os.path.join("/DL/member/zy/xiaohonghu_label_2/", date)
        save_dir = os.path.join("/DL/member/zhangpin/xiaohongshu/", date)
        behave_dir = os.path.join("/DL/member/st/xiaohongshu/user_behave_by_day/", date)
        active_dir = os.path.join("/DL/member/st/xiaohongshu/user_hour_active/", date)
        data = DataProcess(data_dir, save_dir)
        event_df = data.process_event()
        user_df = data.process_user(behave_dir, active_dir)
        device_df = data.process_device()
        ip_df = data.process_ip()
        note_df = data.process_note()
