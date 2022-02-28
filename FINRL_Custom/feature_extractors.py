from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    

class RNNFeatureExtractor(BaseFeaturesExtractor):
    """
    A feature extractor to be used with RL algorithms in stable-baselines
    """

    def __init__(self, *args, observation_space=None, input_size=None, hidden_size=128, rnn_type="lstm", dropout=0, num_layers=2, bidirectional=False):
        features_dim = int(hidden_size * 2) if bidirectional else hidden_size
        super(RNNFeatureExtractor, self).__init__(observation_space, features_dim=features_dim)
        assert rnn_type in ["lstm", "gru"], "rnn_type should be either lstm or gru"
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional)

    def forward(self, x):
        x, _ = self.rnn(x)
        return x.mean([1])


