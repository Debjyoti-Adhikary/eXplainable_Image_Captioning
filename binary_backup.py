import torch
import torch.nn as nn
import torchvision.models as models
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

        def lstm_rec(self,inputs, states, m_len, caption, threshold=0.5):

            if caption[-1] == 12 or caption[-1] == 1 :
                return

            if m_len <= 0 or len(caption) > 20 :
                print(m_len)
                print(f'{len(caption)} :::: {caption}')
            else :
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
                # print(targets[0].item())
                # print(out)
                # for temp in out:
                #     print(temp)
                # print(f'out = {out}') targets = []
                    if (target_score.item() - candidates.values[0][item]) < threshold:
                        targets.append(candidates.indices[0][item])
                m_len = m_len - 1
                for target in targets:
                    caption.append(target.item())
                    inputs = self.word_embeddings(torch.unsqueeze(target, dim=-1)).unsqueeze(1)
                    targets.remove(target)
                    lstm_rec(self, inputs, states, m_len, caption)

        lstm_rec(self,inputs, states, max_len, [0], threshold=0.5)

        # for i in range(max_len):
        #     lstm_out, states = self.lstm(inputs, states)
        #     # print(states)
        #     lstm_out = lstm_out.squeeze(1)
        #     lstm_out = lstm_out.squeeze(1)
        #     # state_times.append(states)
        #     outputs = self.linear(lstm_out)
        #     print(len(states))
        #     # Get maximum probabilities
        #     predicted_scores.append(torch.topk(outputs,5).values)
        #     predicted_indices.append(torch.topk(outputs,5).indices)
        #     target = outputs.max(1)[1]
        #     print(f'target = {target}')
        #     top_prob_words = torch.topk(outputs, 10)
        #     max_prob = torch.topk(outputs, 1)[0]
        #
        #     for variable in range(1,len(top_prob_words.values[0])):
        #         # print(f'{variable} --- {i}')
        #         temp = top_prob_words.values[0][variable]
        #         if (max_prob - temp) < threshold:
        #             print(target)
        #             print(temp)
        #             # print(f'CURRENT : {temp} -- HIGHEST PROBABILITY : {max_prob}')
        #             current_word_list.append(target.item())
        #             print(f'{variable} --- {i}')
        #             print(predicted_sentence)
        #             sentence_till_now.append(predicted_sentence.copy())
        #             caption_possibilities.append(temp)
        #             possible_caption_indices.append(top_prob_words.indices[0][variable])
        #             time_stamps.append(i)
        #             state_times.append(states)
        #
        #     # Append result into predicted_sentence list
        #     predicted_sentence.append(target.item())
        #
        #     prev_word = target
        #     # Update the input for next iteration
        #     inputs = self.word_embeddings(target).unsqueeze(1)
        #     # print(f'{inputs} --- {target}')
        #
        # print(f'{len(time_stamps)}.......{len(state_times)}........{len(possible_caption_indices)}.......{len(caption_possibilities)}')
        # print(f'{len(sentence_till_now)}.......{len(current_word_list)}')
        # # print(caption_possibilities)
        # # print(possible_caption_indices)
        # # print(time_stamps)
        # self.binary_sampling(time_stamps, state_times, possible_caption_indices, caption_possibilities,
        #                 sentence_till_now, current_word_list)
        # return predicted_sentence,predicted_scores,predicted_indices

    def binary_sampling(self, time_stamps,state_t,possible_caption_indices,caption_possibilities, sentence_till_now,current_word_list):

        for i in range(1,len(time_stamps)):
            print(i)
            # print(sentence_till_now[i])
            # print(time_stamps[i])
            print(possible_caption_indices[i])
            s_predicted_scores = []
            s_predicted_sentence = []
            s_predicted_indices = []
            print(inputs.shape)
            inputs = self.word_embeddings(possible_caption_indices[i]).unsqueeze(-1)
            print(inputs.shape)
            states = state_t[i]
            print(len(states))
            for word in range(time_stamps[i],20):
                lstm_out, states = self.lstm(inputs, states)

                lstm_out = lstm_out.squeeze(1)
                lstm_out = lstm_out.squeeze(1)

                outputs = self.linear(lstm_out)

                    # Get maximum probabilities
                s_predicted_scores.append(torch.topk(outputs, 5).values)
                s_predicted_indices.append(torch.topk(outputs, 5).indices)
                target = outputs.max(1)[1]

                    # Append result into predicted_sentence list
                s_predicted_sentence.append(target.item())
                print(s_predicted_sentence)
                    # Update the input for next iteration

                inputs = self.word_embeddings(target).unsqueeze(1)


