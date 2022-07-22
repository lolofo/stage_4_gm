import matplotlib.figure
import numpy as np
import torch
from transformers import BertTokenizer
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

sns.set_theme()

# the tokenizer
tk = BertTokenizer.from_pretrained('bert-base-uncased')


##############################################
## the raw attention class >> define the main function to manipulate the attention
## define it as a class will help us to regroup some technics
##############################################

# some exceptions for the class
# this will help us to make the different steps in the good order

class HeadsAgregationError(Exception):
    pass


class AdjMatrixError(Exception):
    pass


class NoneItemError(Exception):
    pass


class RawAttention:
    """ RawAttention

    Attributes
    ----------

    input_ids : torch.tensor
        the ids of the sentence

    attention_mask : torch.tensor
        the attention mask of the sentence (to spot the [PAD] tokens)

    tokens : list<str>
        list of the tokens in the sentence (each token is a string)

    attention_tensor : torch.tensor
        the attention for the model, it is a torch tensor of the shape (1, n_layer, n_head, n_tokens, n_tokens)

    att_tens_agr : torch.tensor
        the attention tensor after that the heads were agregated
        shape : (n_layer, n_tokens, n_tokens)

    adj_mat : np.array
        the adjacency matrix for the attention graph built on the att_tens_agr
        shape : ((n_layer+1)*n_tokens, (n_layer+1)*n_tokens)

    attention_graph : nx.Digraph
        the attention graph representing all the connections between the words from one layer to another.


    Methods
    -------



    """

    @torch.no_grad()
    def __init__(self, model, input_ids, attention_mask, test_mod=True,
                 *args, **kwargs):

        # the base values
        self.input_ids = input_ids
        self.attention_mask = attention_mask

        # the corresponding tokens
        mask = attention_mask[0, :].detach().numpy() == 1
        self.tokens = tk.convert_ids_to_tokens(input_ids[0, mask])

        # the attention tensor
        self.attention_tensor = None
        outputs = model.bert(input_ids=input_ids,
                             attention_mask=attention_mask,
                             *args, **kwargs)

        buff = outputs.attentions
        res = torch.stack(buff, dim=1)

        # remove the padding tokens
        mask = attention_mask[0, :].detach().numpy() == 1
        res = res[:, :, :, mask, :]
        res = res[:, :, :, :, mask]
        self.attention_tensor = res.detach().clone()  # >> fastest way to clone a tensor

        ## >> start test for the attention tensor
        if test_mod:
            print("test passed : ", end='')
            passed = True
            for n in range(len(buff)):
                for n_head in range(12):
                    for x in range(input_ids.shape[1]):
                        for y in range(input_ids.shape[1]):
                            if mask[x] == 1 and mask[y] == 1:
                                if buff[n][0, n_head, x, y] != self.attention_tensor[0, n, n_head, x, y]:
                                    passed = False
                                    break

            if passed:
                print(u'\u2713')
            else:
                print("x")
        ## >> end test
        self.heads_agr = False  # did we proceed the heads agregation
        self.att_tens_agr = None  # the tensor with heads agregation (which layer is significant)
        self.att_tens_lay_agr = None  # the tensor with layer agregation (which head is significant)

        # attention graph stuff
        self.adj_mat = None
        self.adj_mat_done = False
        self.label = None
        self.attention_graph = None
        self.set_gr = False

    # print and have some informations about the attention structure !
    def __str__(self):
        result = ""
        result += f">> the tokens : {self.tokens}" + "\n"
        result += f">> graph set up : {self.set_gr}" + " "
        if self.set_gr:
            result += ">> ready for inference !"
        else:
            result += ">> NOT ready for inference !"
        return result

    # have a quick access to some things
    def __getitem__(self, item):
        if item == "attention_graph":
            return self.attention_graph.copy()
        elif item == "attention_tensor":
            return self.attention_tensor.detach().clone()
        elif item == "attention_tensor_agreg":
            return self.att_tens_agr.detach().clone()
        else:
            raise NoneItemError(
                "please select item in range [attention_graph, attention_tensor, attention_tensor_agreg]"
            )

    ########################
    ### heads agregation ###
    ########################
    @torch.no_grad()
    def heads_agregation(self,
                         heads_concat: bool = True,
                         agr_type="avg",
                         num_head: int = -1,
                         prune_mask=None
                         ):
        """ Agregation of the different attention heads

        :param heads_concat : if true we proceed head agregation (combination of the different heads)
        :param agr_type : what agregation should we proceed
        :param num_head : if we do not proceed any agregation, what head should we use
        :param prune_mask: this mask will contain on every position what head are usefull
        """
        self.heads_agr = True

        if heads_concat:
            if num_head > 0:
                warnings.warn("The heads number is useless since you want to proceed heads agregation")
        else:
            if num_head < 0 or num_head > 11:
                raise HeadsAgregationError("the attention head you wan't to select doesn't exists !")

        if heads_concat:
            # proceed the head agregation
            n_layer = self.attention_tensor.shape[1]
            n_head = self.attention_tensor.shape[2]
            # one attention tensor per layer >> agregation of the heads
            self.att_tens_agr = np.zeros((n_layer, len(self.tokens), len(self.tokens)))
            if agr_type == "avg":
                self.att_tens_agr = self.attention_tensor[0, :, :, :, :].mean(dim=1) # mean over the heads
            elif agr_type == "max":
                self.att_tens_agr = self.attention_tensor[0, :, :, :, :].max(dim=1)[0] # max over the heads
            elif agr_type == "cls":
                # TODO create the attention for the CLS map
                pass



        else:
            # clone just the tensor without the gradient
            # here the gradient is not usefull
            self.att_tens_agr = self.attention_tensor[0, :, num_head, :, :].detach().clone()

    ################################
    ### defining the graph tools ###
    ################################
    @torch.no_grad() # private method.
    def _create_adj_matrix(self):

        if not self.heads_agr:
            raise HeadsAgregationError("You can't create adj matrix without proceeding heads agregation")

        length = len(self.tokens)
        n_layers, _, _ = self.att_tens_agr.shape  # number of attention heads
        self.adj_mat = np.zeros(((n_layers + 1) * length, (n_layers + 1) * length))

        # the labels -> the name of each node to know where it is.
        self.label = {}

        for i in range(n_layers + 1):
            if i == 0:
                for u in range(length):
                    # non contextual embeddings
                    buff = "Layer_" + str(i) + "_" + str(u)
                    self.label[buff] = u
            else:
                for u in range(length):
                    k_u = length * i + u
                    buff = "Layer_" + str(i) + "_" + str(u)
                    self.label[buff] = k_u
                    for v in range(length):
                        k_v = length * (i - 1) + v
                        # one the next line >> i-1 because of how we count the layers
                        # adj_mat[i,j] >> edge from i to j
                        self.adj_mat[k_u][k_v] = self.att_tens_agr[i - 1][u][v].item()

        self.adj_mat_done = True

    def _create_attention_graph(self):
        """Creation of a networkx Digraph based on the adj_matrix.

        /!\ >> the attention graph has sens only for the heads agregation.
        for the the layer agregation it has just no sens
        """

        # creation of the graph from the adjacency matrix then :
        if not self.adj_mat_done:
            raise AdjMatrixError("You can't create graph without adj matrix")

        g = nx.from_numpy_matrix(self.adj_mat, create_using=nx.DiGraph())

        for i in np.arange(self.adj_mat.shape[0]):
            for j in np.arange(self.adj_mat.shape[1]):
                # setting the capacity attribute for the maximum flow problem
                nx.set_edge_attributes(g, {(i, j): self.adj_mat[i, j]}, 'capacity')

        # the graph is also created to have capacities so we can perform max flow problem on it
        self.attention_graph = g.copy()

    ########################################################
    ## combine the previous functions to set up the graph ##
    ########################################################
    def set_up_graph(self, num_head=-1, heads_concat=True, agr_type="avg", prune_mask=None):
        """ The different step to set up the graph

        - proceed the agregation of the heads
        - create the adjacency matrix
        - create the graph thanks to the matrix
        """
        self.heads_agregation(heads_concat=heads_concat,
                              num_head=num_head,
                              agr_type=agr_type,
                              prune_mask=prune_mask)

        self._create_adj_matrix()
        self._create_attention_graph()

        # the graph is now set up !
        self.set_gr = True

    # create the figure
    def draw_attention_graph(self,
                             graph_width: int = 20,
                             n_layers: int = 12
                             ) -> matplotlib.figure.Figure:
        """ Draw the attention graph

        :param graph_width : the size of the final graph
        :param n_layers : number of multi-head-attention layer in the model (in bert it is 12)
        """

        # deal with different warnings
        if not self.set_gr:
            warnings.warn("you didn't set up the graph so we proceed it with heads agregation")
            # we must find a way to concat the different heads
            self.set_up_graph(heads_concat=True)

        pos = {}
        label_pos = {}
        length = len(self.tokens)
        for i in np.arange(n_layers + 1):
            for k_f in np.arange(length):
                pos[i * length + k_f] = ((i + 0.4) * 2, length - k_f)
                label_pos[i * length + k_f] = (i * 2, length - k_f)

        index_to_labels = {}
        for key in self.label:
            index_to_labels[self.label[key]] = self.tokens[int(key.split("_")[-1])]
            if self.label[key] >= length:
                # only label the labels on the left side
                index_to_labels[self.label[key]] = ''

        fig = plt.figure(figsize=(graph_width, graph_width))

        #  label=index_to_labels
        nx.draw_networkx_nodes(self.attention_graph, pos, node_color='green', node_size=50)
        nx.draw_networkx_labels(self.attention_graph, pos=label_pos, labels=index_to_labels, font_size=10)

        # draw the edges
        all_weights = []
        for (node1, node2, data) in self.attention_graph.edges(data=True):
            all_weights.append(data['weight'])

        unique_weights = list(set(all_weights))

        for weight in unique_weights:
            weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in self.attention_graph.edges(data=True) if \
                              edge_attr['weight'] == weight]

            w = weight
            width = w
            nx.draw_networkx_edges(self.attention_graph, pos, edgelist=weighted_edges, width=width,
                                   edge_color='darkred')

        return fig
