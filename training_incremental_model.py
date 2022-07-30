import torch
import torch.nn as nn
from model import EncoderCNN, DecoderRNN
import os

class Incremental_EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet152 and replace top fc layer."""
        super(Incremental_EncoderCNN, self).__init__()

        encoder_file = 'encoder-1.pkl'

        self.encoder = EncoderCNN(embed_size)
        # self.encoder.eval()

        # self.encoder.to(device)

        # self.encoder.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models', encoder_file)))

        for module in self.encoder.parameters():
            module.requires_grad = False

        network_modules = list(self.encoder.children())
        resnet_modules = list(network_modules[0].children())
        # print(len(resnet_modules))
        resnet_modules.pop(7)
        # resnet_modules.pop(6)
        # resnet_modules.pop(5)
        # print(len(resnet_modules))
        self.encoder = nn.Sequential(*resnet_modules)
        self.embed = nn.Linear(1024,embed_size)


    def forward(self, images):
        """Extract the image feature vectors."""

        features = self.encoder(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


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
        decoder_file = 'decoder-1.pkl'
        # embedding layer that turns words into a vector of a specified size
        # self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        hidden_size = 512
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
        # self.decoder.eval()

        self.decoder.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models', decoder_file)))
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)


    def forward(self, features, captions):
        """Extract the image feature vectors."""

        captions = captions[:,:-1]
        embeds = self.decoder.word_embeddings(captions)

        # Concatenating features to embedding
        # torch.cat 3D tensors
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)

        lstm_out, hidden = self.decoder.lstm(inputs)
        outputs = self.decoder.linear(lstm_out)

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
        predicted_scores = []
        predicted_indices = []
        for i in range(max_len):

            lstm_out, states = self.decoder.lstm(inputs, states)

            lstm_out = lstm_out.squeeze(1)
            lstm_out = lstm_out.squeeze(1)

            #
            # lstm_out.backward()
            # score_max_index = lstm_out.argmax()
            # score_max = lstm_out[0, score_max_index]
            # score_max.backward()
            #
            outputs = self.decoder.linear(lstm_out)

            # Get maximum probabilities
            # print(torch.topk(outputs,30))
            predicted_scores.append(torch.topk(outputs,10).values)
            predicted_indices.append(torch.topk(outputs,10).indices)
            target = outputs.max(1)[1]
            # print(outputs.max(1))
            # print(outputs.max(1)[1])
            # word = data_loader.dataset.vocab.idx2word[idx]
            # Append result into predicted_sentence list
            # print(target)
            predicted_sentence.append(target.item())

            # print(target.item())
            # Update the input for next iteration
            inputs = self.decoder.word_embeddings(target).unsqueeze(1)
            # print(self.word_embeddings(target).unsqueeze(1).max)

        return predicted_sentence,predicted_scores,predicted_indices

        # predicted_sentence = []
        #
        # for i in range(max_len):
        #
        #     lstm_out, states = self.decoder.lstm(inputs, states)
        #
        #     lstm_out = lstm_out.squeeze(1)
        #     lstm_out = lstm_out.squeeze(1)
        #     # lstm_out.backward()
        #     # score_max_index = lstm_out.argmax()
        #     # score_max = lstm_out[0, score_max_index]
        #     # score_max.backward()
        #     #
        #     outputs = self.decoder.linear(lstm_out)
        #
        #     # Get maximum probabilities
        #     target = outputs.max(1)[1]
        #
        #     # Append result into predicted_sentence list
        #     predicted_sentence.append(target.item())
        #
        #     # Update the input for next iteration
        #     inputs = self.decoder.word_embeddings(target).unsqueeze(1)
        #
        # return predicted_sentence