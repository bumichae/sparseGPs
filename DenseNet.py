class DenseNet(nn.Module):
    """
    Simple module implementing a feedforward neural network.
    You can use this model as a reference/baseline for calibration
    in the normal neural network case.
    """

    def __init__(self, in_features: int, hidden_features: typing.Tuple[int, ...], out_features: int):
        """
        Create a normal NN.

        :param in_features: Number of input features
        :param hidden_features: Tuple where each entry corresponds to a hidden layer with
            the corresponding number of features.
        :param out_features: Number of output features
        """
        super().__init__()

        feature_sizes = (in_features,) + hidden_features + (out_features,)
        num_affine_maps = len(feature_sizes) - 1
        self.layers = nn.ModuleList([
            nn.Linear(feature_sizes[idx], feature_sizes[idx + 1], bias=True)
            for idx in range(num_affine_maps)
        ])
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_features = x

        for idx, current_layer in enumerate(self.layers):
            new_features = current_layer(current_features)
            if idx < len(self.layers) - 1:
                new_features = self.activation(new_features)
            current_features = new_features

        return current_features

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.forward(x)
        return pred
    