import networkx as nx
import warnings

from raw_attention import RawAttention


def attention_flow_max(
        raw_attention_inst: RawAttention,
        out_layer: int = 11):
    """ Find the flow max between the output and the input
    - we will only search for the flow max between the CLS token and the inputs tokens
    """
    if not raw_attention_inst.set_gr:
        # set up the graph
        warnings.warn("The graph isn't set up we will then set it up")
        raw_attention_inst.set_up_graph()

    nb_tokens = len(raw_attention_inst.tokens)
    s_label = "Layer_" + str(out_layer) + "_0"  # label of the CLS token (the source node in our graph)
    s = raw_attention_inst.label[s_label]  # source node
    res = []

    for i in range(nb_tokens):
        t_label = "Layer_0_" + str(i)
        t = raw_attention_inst.label[t_label]
        # the maximum flow
        #
        flow_val, _ = nx.maximum_flow(raw_attention_inst.attention_graph, s, t)
        res.append(flow_val)

    return res
