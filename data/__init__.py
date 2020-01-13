# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 3:47 下午
# @Author  : zhangpin
# @Email   : zhangpin@geetest.com
# @File    : __init__.py
# @Software: PyCharm

from .embedding_table import EmbeddingTable

embed_file_path = "./embedding_files"
embedding_table = EmbeddingTable(embed_file_path)
