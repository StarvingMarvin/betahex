
class CommonModel:

    def __init__(self, board_size=13, input_features=None, layer_dim_5=60, layer_dims_3=None):
        self.board_size = 13
        self.layer_dim_5 = layer_dim_5
        self.layer_dims_3 = layer_dims_3 or [60]

        self.input_features = input_features or []

        self.feature_dim = (3 * board_size // 2, board_size)
