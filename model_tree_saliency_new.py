import torch
import torch.nn as nn
import torchvision.models as models
import copy
from utils import clean_sentence
from torchsummary import summary

class EncoderCNN_b(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet152 and replace top fc layer."""
        super(EncoderCNN_b, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(True)

        modules = list(resnet.children())[:-1]
        input_length = resnet.fc.in_features
        #resnet.fc.in_features #64 #256 #512 #1024 #resnet.fc.in_features
        print(len(modules),input_length)
        self.resnet = nn.Sequential(*modules)

        # print(summary(self.resnet))
        self.embed = nn.Linear(input_length,embed_size)
        # self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.init_weights()

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


class DecoderRNN_b(nn.Module):
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
        super(DecoderRNN_b, self).__init__()
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

    def sample(self, inputs, states=None, max_len=20, threshold = 0.5):
        """
        Greedy search:
        Samples captions for pre-processed image tensor (inputs)
        and returns predicted sentence (list of tensor ids of length max_len)
        """

        def lstm_rec(self,inputs, states, m_len,all_candidates, output_scores, caption, current,current_output,current_candidates, threshold=0.5):

            if current!=None:
                caption.append(current)
                output_scores.append(current_output)
                all_candidates.append(current_candidates)
            # print(output_scores)

            if caption and (caption[-1] == 12 or caption[-1] == 1):
                # print(f'caption = {clean_sentence(caption)}')
                yield {clean_sentence(caption) : [output_scores,all_candidates]}

            # if m_len <= 0 or len(caption) > 20 :
            #     print(m_len)
            #     # print(f'{len(caption)} :::: {caption}')
            if m_len > 0 and len(caption) < 20:
                lstm_out, states = self.lstm(inputs, states)
                lstm_out = lstm_out.squeeze(1)
                lstm_out = lstm_out.squeeze(1)
                outputs = self.linear(lstm_out)

                candidates = torch.topk(outputs, 5)
                # Get maximum probabilities
                ## This was the inital target (it will always be executed first)
                target_index = outputs.max(1)[1]
                target_score = outputs.max(1)[0]
                targets = []

                ## this loop is for generating possible targets (that lie in epsilon range)
                ## if we do not include the max element , targets remains empty and the following loop doesn't get
                ## executed and hence function wont call itself
                for item in range(0,len(candidates.values[0])):
                    ## check for close scores here i.e. if score is within epsilon from top
                    if (target_score.item() - candidates.values[0][item].item()) < threshold:
                        targets.append(candidates.indices[0][item])
                m_len = m_len - 1
                for target in targets:
                    # caption.append(target.item())
                    inputs = self.word_embeddings(torch.unsqueeze(target, dim=-1)).unsqueeze(1)
                    targets.remove(target)
                    # output_scores.append(outputs)
                    yield from lstm_rec(self, inputs, states, m_len,all_candidates.copy(), output_scores.copy(), copy.deepcopy(caption), target.item(),outputs,candidates, threshold)

        return lstm_rec(self,inputs, states, max_len,[], [], [], None, None,None, threshold=0.7)

    # lstm_rec(self, inputs, states, m_len,all_candidates, output_scores.copy(), copy.deepcopy(caption), target.item(),outputs,candidates[0], threshold)
    # (self,inputs, states, m_len,all_candidates, output_scores, caption, current,current_output,current_candidates, threshold=0.5)