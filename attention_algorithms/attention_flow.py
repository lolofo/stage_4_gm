"""
Attention flow problem :
    - look at the attention trough the layers as a flow maximization problem
    - in this file, when we talk about the length it is the len of the sentence we are looking at

/!\ we must deal with the attention mask also in our functions
"""

import torch
import networkx as nx
import numpy as np

from raw_attention import heads_transf


def attention_flows():
    pass
