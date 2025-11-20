import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()

        def ConvBN(in_c, out_c, k, s):
            return nn.Sequential(
                nn.Conv(in_c, out_c, k, stride=s, bias=True, device=device, dtype=dtype),
                nn.BatchNorm2d(out_c, device=device, dtype=dtype),
                nn.ReLU(),
            )

        self.c1 = ConvBN(3,   16, 7, 4)   # 32x32 -> 8x8
        self.c2 = ConvBN(16,  32, 3, 2)   # 8x8   -> 4x4

        self.r1a = ConvBN(32, 32, 3, 1)
        self.r1b = ConvBN(32, 32, 3, 1)

        self.c3 = ConvBN(32,  64, 3, 2)   # 4x4 -> 2x2
        self.c4 = ConvBN(64, 128, 3, 2)   # 2x2 -> 1x1

        self.r2a = ConvBN(128, 128, 3, 1)
        self.r2b = ConvBN(128, 128, 3, 1)

        self.head = nn.Sequential(
            nn.Flatten(),                                 # (N,128,1,1) -> (N,128)
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype),
        )

    def forward(self, x):
        # x: (N, 3, 32, 32)
        x = self.c1(x)
        x = self.c2(x)

        # Residual block 1
        skip = x
        x = self.r1a(x)
        x = self.r1b(x)
        x = x + skip

        x = self.c3(x)
        x = self.c4(x)

        # Residual block 2
        skip = x
        x = self.r2a(x)
        x = self.r2b(x)
        x = x + skip

        return self.head(x)



class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
