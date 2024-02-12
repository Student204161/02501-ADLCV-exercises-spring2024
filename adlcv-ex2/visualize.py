

import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT
import cv2
from einops import rearrange, repeat
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_activation(name):
    """Hook to get the activation of a (intermediate) layer. Used to visualize the feature maps. 
    args:
        name: name of the layer
    """
    def hook(model, input, output):
        hooks[name] = output.detach()
    return hook

def plot_attention_map(original_img, att_map, alpha):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 16))

    #normalize att_map
    norm_att_map = (att_map - torch.min(att_map))/(torch.max(att_map) - torch.min(att_map))
    #resize
    mask = cv2.resize(norm_att_map.numpy(), (32,32))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)      

    beta = (1.0 - alpha)
    attention_img = cv2.addWeighted(np.uint8(original_img.numpy()*255), alpha, heatmap, beta, 0.0)

    

    ax1.set_title('Original')
    ax2.set_title('Attention Map (Max fusion)')
    ax3.set_title('Attention Map Last Layer')
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(heatmap)
    _ = ax3.imshow(attention_img)

def attention_map_from_kq(keys,queries):
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5

    # Rearrange keys, queries and values 
    # from batch_size x seq_len x embed_dim to (batch_size x num_head) x seq_len x head_dim
    keys = rearrange(keys, 'b s (h d) -> (b h) s d', h=num_heads, d=head_dim)
    queries = rearrange(queries, 'b s (h d) -> (b h) s d', h=num_heads, d=head_dim)


    attention_logits = torch.matmul(queries, keys.transpose(1, 2))
    attention_logits = attention_logits * scale
    attention = F.softmax(attention_logits, dim=-1)
    
    attention_heads = attention[:,64,:64] #discard CLS token to get  vector of size 64
    attention_heads = attention_heads.reshape(attention.shape[0],8,8) #reshape to 8x8

    attention_map = np.copy(attention_heads[0])
    for i in range(attention.shape[0]):
        attention_map = np.maximum(attention_map,attention_heads[i]) #use max merge of attention heads

    return attention_map
    

import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def grad_rollout(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):                
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()
        output = self.model(input_tensor)
        category_mask = torch.zeros(output.size())
        category_mask[:, category_index] = 1
        loss = (output*category_mask).sum()
        loss.backward()

        return grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio)




if __name__ == "__main__":
    
    model_path ='model.pth'
    trainset = torch.load("trainset.pt")
    testset = torch.load("testset.pt")
    image_size=(32,32)
    patch_size=(4,4)
    channels=3
    embed_dim=256
    num_heads=4
    num_layers=2
    num_classes=2
    pos_enc='learnable'
    pool='cls'
    dropout=0.3
    fc_dim=None
    num_epochs=20
    batch_size=1
    lr=1e-4
    warmup_steps=625
    weight_decay=1e-3
    gradient_clipping=1

    train_transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes
    )
    criterion = torch.nn.CrossEntropyLoss()

    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights)
    model.eval()
    hooks = {} # dictionary to store intermidate layers
    
    model.transformer_blocks[-1].attention.k_projection.register_forward_hook(get_activation("k_proj")) #we want to visualize the attention map of the last layer
    model.transformer_blocks[-1].attention.q_projection.register_forward_hook(get_activation("q_proj")) #we want to visualize the attention map of the last layer

    #make tensor out of first 8 images in trainset
    img_index = 21
    input_img, input_label = trainset.dataset[img_index]

    input_data = train_transform(input_img)
    input_data = input_data.unsqueeze(0)

    output = model(input_data)
    keys = hooks["k_proj"].detach()
    queries = hooks["q_proj"].detach()

    attention_map = attention_map_from_kq(keys, queries)

    plot_attention_map(input_img.permute(2,1,0), attention_map, 0.6) #not resized attention map...
    plt.show()
    print('hey')
    plot_attention_map(input_img.permute(2,1,0), attention_map, 0.8) #not resized attention map...
    plt.show()
    print('hey2')
    ### if performing rollout, we should register hooks for all layers
    # for indx, transformer_block in enumerate(model.transformer_blocks):
    #     hooks[indx] = transformer_block.attention.register_forward_hook(get_activation(str(indx)))

    ##initialize tensor to store attention maps
    # attentions = []
    # for attention_indx in range(num_layers):
    #     attentions.append(hooks[str(attention_indx)])
    ##convert into tensor
    #attentions = torch.stack(attentions)
    