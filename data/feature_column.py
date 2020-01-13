# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 4:32 下午
# @Author  : zhangpin
# @Email   : zhangpin@geetest.com
# @File    : feature_column
# @Software: PyCharm

import six
from tensorflow import feature_column


def numeric_column(key, shape=(1,), normalizer_fn=None):
    column = feature_column.numeric_column(key=key, shape=shape, normalizer_fn=normalizer_fn)
    return column


def bucketized_column(key, shape, boundaries=None):
    numeric_col = numeric_column(key, shape)
    return feature_column.bucketized_column(numeric_col, boundaries=boundaries)


def categorical_column(key, vocabulary_size=None,
                       vocabulary_list=None,
                       vocabulary_file=None,
                       num_oov_buckets=0):
    if vocabulary_size:
        categorical_col = feature_column.categorical_column_with_identity(key, vocabulary_size)
        return categorical_col
    elif vocabulary_list:
        assert isinstance(vocabulary_list[0], six.string_types), "Vocabulary must be sequence of string"
        categorical_col = feature_column.categorical_column_with_vocabulary_list(key, vocabulary_list, num_oov_buckets)
        return categorical_col
    elif vocabulary_file:
        categorical_col = feature_column.categorical_column_with_vocabulary_file(key, vocabulary_file, num_oov_buckets)
        return categorical_col


def onehot_column(key, vocabulary_size=None, vocabulary_list=None):
    return feature_column.indicator_column(categorical_column(key, vocabulary_size, vocabulary_list))


def embedding_column(key, dimension, vocabulary_size=None,
                     vocabulary_file=None,
                     vocabulary_list=None,
                     num_oov_buckets=0):
    return feature_column.embedding_column(categorical_column(key, vocabulary_size, vocabulary_list, vocabulary_file, num_oov_buckets),
                                           dimension=dimension)
