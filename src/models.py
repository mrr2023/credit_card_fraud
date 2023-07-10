import torch


class MLPClassifier(torch.nn.Module):
    def __init__(self, n_input, layers, dropout=False):
        super().__init__()

        L = []

        for layer_idx, layer in enumerate(layers):
            if dropout != False:
                L.append(torch.nn.Dropout(dropout))
                
            L.append(torch.nn.Linear(n_input, layer))

            # Don't add batch normalization to output of final layer
            if layer_idx != len(layers) - 1:
                L.append(torch.nn.BatchNorm1d(layer))

            L.append(torch.nn.ReLU())
            n_input = layer

        self.network = torch.nn.Sequential(*L)


    def forward(self, x):
        x = self.network(x).float()
        return x