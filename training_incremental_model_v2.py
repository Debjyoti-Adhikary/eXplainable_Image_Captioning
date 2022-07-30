import torch
import torch.nn as nn
import torchvision.models as models
import os

class Incremental_EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet152 and replace top fc layer."""
        super(Incremental_EncoderCNN, self).__init__()

        encoder_file = 'encoder-1.pkl'

        resnet = models.resnet152(pretrained=True)

        # for param in resnet.parameters():
        #     param.requires_grad_(True)

        modules = list(resnet.children())[:-1]

        input_length = 1024 #resnet.fc.in_features #64 #256 #512 #1024 #resnet.fc.in_features
        modules.pop(7)
        print(len(modules),input_length)
        self.resnet = nn.Sequential(*modules)
        # print(summary(self.resnet))
        self.resnet.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models', encoder_file)))
        self.embed = nn.Linear(input_length,embed_size)
        for parameter in self.resnet.parameters():
            parameter.requires_grad_(False)
        # self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        # self.init_weights()

    def forward(self, images):
        """Extract the image feature vectors."""

        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

    def init_weights(self):
        """Initialize the weights."""
        self.embed.weight.data.normal_(0.0, 0.02)
        self.embed.bias.data.fill_(0)


class Incremental_DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Set the hyper-parameters and build the layers.
        Parameters
        ----------
        - embed_size  : Dimensionality of image and word embeddings
        - hidden_size : number of features in hidden state of the RNN decoder
        - vocab_size  : The size of vocabulary or output size
        - num_layers  : Number of layers

        """
        super(Incremental_DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        # The LSTM takes embedded vectors as inputs
        # and outputs hidden states of hidden_size
        self.lstm = nn.LSTM(input_size = embed_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)

        # the linear layer that maps the hidden state output dimension
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def forward(self, features, captions):
        """Extract the image feature vectors."""

        captions = captions[:,:-1]
        embeds = self.word_embeddings(captions)

        # Concatenating features to embedding
        # torch.cat 3D tensors
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)

        lstm_out, hidden = self.lstm(inputs)
        outputs = self.linear(lstm_out)

        return outputs

    def init_weights(self):
        """Initialize weights."""

        self.word_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def sample(self, inputs, states=None, max_len=20):
        """
        Greedy search:
        Samples captions for pre-processed image tensor (inputs)
        and returns predicted sentence (list of tensor ids of length max_len)
        """

        predicted_sentence = []

        for i in range(max_len):

            lstm_out, states = self.lstm(inputs, states)

            lstm_out = lstm_out.squeeze(1)
            lstm_out = lstm_out.squeeze(1)

            #
            # lstm_out.backward()
            # score_max_index = lstm_out.argmax()
            # score_max = lstm_out[0, score_max_index]
            # score_max.backward()
            #
            outputs = self.linear(lstm_out)

            # Get maximum probabilities
            target = outputs.max(1)[1]

            # Append result into predicted_sentence list
            predicted_sentence.append(target.item())

            # Update the input for next iteration
            inputs = self.word_embeddings(target).unsqueeze(1)

        return predicted_sentence