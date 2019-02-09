# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 19:19:12 2019

@author: nshah
non-sparse pivot convert it to sparse representation

"""

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

from pandas.api.types import CategoricalDtype
from mlxtend.frequent_patterns import apriori


file1 = 'mbatest.txt'

frame = pd.read_csv(file1, sep=",",nrows=5000000)
frame['QTY'] = 1

trx_id  = frame['TRX_ID']
product = frame['PRODUCT']
qty     = frame['QTY']

TRX_ID_u   = sorted(frame.TRX_ID.unique())
PRODUCT_u  = sorted(frame.PRODUCT.unique())


TRX_ID_c   = CategoricalDtype(sorted(frame.TRX_ID.unique()), ordered=True)
PRODUCT_c  = CategoricalDtype(sorted(frame.PRODUCT.unique()), ordered=True)

row = frame.TRX_ID.astype(TRX_ID_c).cat.codes
col = frame.PRODUCT.astype(PRODUCT_c).cat.codes
sparse_matrix = csr_matrix((frame["QTY"], (row, col)), \
                           shape=(TRX_ID_c.categories.size, PRODUCT_c.categories.size))

coo1 = sparse_matrix.tocoo(copy=True)

df = pd.SparseDataFrame(coo1, \
                         index=TRX_ID_c.categories, \
                         columns=PRODUCT_c.categories, \
                         default_fill_value=0)

frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

#from mlxtend.frequent_patterns import association_rules
#res = association_rules(frequent_itemsets, support_only=True, min_threshold=0)