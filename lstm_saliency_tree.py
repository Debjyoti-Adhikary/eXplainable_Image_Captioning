import torch
import torch.nn.functional as F
from utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer, find_other_layer

class Saliency():
    def __init__(self,encoder, decoder,target_layer, input_size =(224,224)):
        self.encoder = encoder
        self.decoder = decoder
        # target_layer = [*encoder.children()][0][0] #[*encoder.children()][-2][-2][-1].conv3 #[*encoder.children()][0][4][0].conv1 #[*decoder.children()][3] [*encoder.children()][-2][-2][-1].conv3
        target_layer = target_layer
        print(target_layer)
        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]  #
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        # device = 'cuda' if next(self.encoder.parameters()).is_cuda else 'cpu'
        # model_input = torch.zeros(1, 3, *(input_size))  # original =  1, 3, *(input_size)
        # model_input = model_input.to(device)
        # # v_features = self.encoder(model_input.to('cuda')).unsqueeze(1)
        # self.decoder.sample(self.encoder(model_input.to('cuda')).unsqueeze(1).to('cuda'))


    def forward(self, raw_input):
        b, c, h, w= raw_input.size() #removed batch size b
        v_features = self.encoder(raw_input.to('cuda')).unsqueeze(1)
        generator = self.decoder.sample(v_features.to('cuda'))
        self.encoder.train()
        self.decoder.train()

        super_dictionary = {}
        for dictionary in generator:
            for key, value in dictionary.items():
                super_dictionary.setdefault(key,value)

        cap_sal_dict = {}
        for keys in super_dictionary:
            # print(f'{keys} + {super_dictionary[keys]}')
            logits = super_dictionary[keys]
            caption = keys
            saliency_maps = []
            for logit in logits:
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                score = logit[:, logit.max(1)[-1]].squeeze()

                score.backward(retain_graph=True)

                gradients = self.gradients['value']
                activations = self.activations['value']

                b, k,_,_ = gradients.size()
                alpha = gradients.view(b, k, -1).mean(2)
                weights = alpha.view(b, k, 1, 1)
                saliency_map = (weights*activations).sum(1, keepdim=True)
                saliency_map = F.relu(saliency_map)
                top_predictions = torch.topk(saliency_map,5)

                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
                saliency_maps.append(saliency_map)

            cap_sal_dict[caption] = saliency_maps
        # saliency_map = torch.mul(100,saliency_map)
        return cap_sal_dict
        # return caption, saliency_maps, logits

    def __call__(self,raw_input):
        return self.forward(raw_input)
