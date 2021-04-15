import torch.nn as nn

class Network(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, numclass, feature_extractor, proj_size=128):
        super(Network, self).__init__()
        self.feature = feature_extractor
        #TODO: Add two layer projection head
        self.proj_head = nn.Sequential(
            nn.Linear(feature_extractor.fc.in_features, feature_extractor.fc.in_features),
            nn.Linear(feature_extractor.fc.in_features, feature_extractor.fc.in_features)
        )

        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)

    def forward(self, input):
        x = self.feature(input)
        x = self.proj_head(x)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self,inputs):
        return self.proj_head(self.feature(inputs))