import torch.nn as nn


class network(nn.Module):

    def __init__(self, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = None
    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass,device):
        if self.fc is None:
            self.fc = nn.Linear(self.feature.fc.in_features, numclass, bias=True)
            self.fc.to(device)
        else:
            weight = self.fc.weight.data
            bias = self.fc.bias.data
            in_feature = self.fc.in_features
            out_feature = self.fc.out_features
            del self.fc 
            self.fc = nn.Linear(in_feature, numclass, bias=True)
            self.fc.weight.data[:out_feature] = weight
            self.fc.bias.data[:out_feature] = bias
            self.fc.to(device)
        self.features_dim = self.fc.in_features
    def reduce(self):
        pass
        

    def feature_extractor(self,inputs):
        return self.feature(inputs)

class reduced_network(nn.Module):

    def __init__(self, feature_extractor,numclass,device):
        super(reduced_network, self).__init__()
        self.conv1 = feature_extractor.feature.conv1
        self.layer1 = feature_extractor.feature.layer1
        self.layer2 = feature_extractor.feature.layer2
        self.layer3 = feature_extractor.feature.layer3
        self.avgpool = feature_extractor.feature.avgpool
        self.fc = feature_extractor.fc
        weight = feature_extractor.fc.weight.data
        bias = feature_extractor.fc.bias.data
        out_feature = feature_extractor.fc.out_features
        self.fc.to(device)
    def forward(self, input):
        # x = self.feature(input)
        x = self.conv1(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # OUT_FEATURES updated to NUMCLASS
    def Incremental_learning(self, numclass,device):
        # if self.fc is None:
        if self.fc: del self.fc
        self.fc = nn.Linear(self.layer1.out_features, numclass, bias=True)
        self.fc.to(device)
        # else:
        #     weight = self.fc.weight.data
        #     bias = self.fc.bias.data
        #     in_feature = self.feature.layer1.out_features
        #     out_feature = self.fc.out_features
        #     del self.fc 
        #     self.fc = nn.Linear(in_feature, numclass, bias=True)
        #     self.fc.weight.data[:out_feature] = weight
        #     self.fc.bias.data[:out_feature] = bias
        #     self.fc.to(device)
        self.features_dim = self.fc.in_features


    def feature_extractor(self,inputs):
        return self.feature(inputs)

"""
class network(nn.Module):

    def __init__(self, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = None
    
    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        if self.fc is None:
            self.fc = nn.Linear(self.feature.fc.in_features, numclass, bias=True)

        else:
            weight = self.fc.weight.data
            bias = self.fc.bias.data
            in_feature = self.fc.in_features
            out_feature = self.fc.out_features

            self.fc = nn.Linear(in_feature, numclass, bias=True)
            self.fc.weight.data[:out_feature] = weight
            self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self,inputs):
        return self.feature(inputs)


class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)

    def forward(self, input):
        x = self.feature(input)
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
        return self.feature(inputs)
"""