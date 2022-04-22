"""
Work on the attention :

    - the attention will be a tensor.
    - shape of the tensor : (batch_size, num_heads, sequence_length, sequence_length)

    - this class is build to work with the `custom_data_set.py`  we created in this project
"""


class AttentionWeight:

    def __init__(self, input_ids, attention_mask, model):
        """
        input : input_ids      --> tensor of the ids of the sentence (batch_size , number of tokens)
                attention_mask --> tensor of the attention of the ids (batch_size, number of tokens)
                model          --> one of the NLI model we created in this project
        """
        pass

    def __call__(self):
        pass

    def visualize(self):
        pass
