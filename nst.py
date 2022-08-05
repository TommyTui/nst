"""
Modified from style_transfer.ipynb of A6 of EECS 498.
"""
import torch
from torch.cuda import FloatTensor as ft
import torchvision
import PIL
from torchvision.models import VGG19_Weights

from nst_helper import *
from style_transfer import *

cnn = torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
cnn.type(ft)
for param in cnn.parameters():
    param.requires_grad = False


def style_transfer(content_image, style_image, content_layer, content_weight,
                   style_layers, style_weights, learning_rate=8e-2, iterations=180, output="output.jpg", verbose=True):
    """
    Run style transfer!

    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    """
    content_img = preprocess(PIL.Image.open(content_image)).type(ft)
    feats = extract_features(content_img, cnn)
    content_target = feats[content_layer].clone()

    style_img = preprocess(PIL.Image.open(style_image)).type(ft)
    feats = extract_features(style_img, cnn)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(feats[idx].clone()))

    img = content_img.clone().type(ft)
    img.requires_grad_()

    optimizer = torch.optim.Adam([img], lr=learning_rate)

    for i in range(iterations):
        with torch.no_grad():
            img.data.clamp_(-2.5, 4)
        optimizer.zero_grad()

        feats = extract_features(img, cnn)

        c_loss = content_loss(
            content_weight, feats[content_layer], content_target)
        s_loss = style_loss(feats, style_layers, style_targets, style_weights)
        loss = c_loss + s_loss
        loss.backward()
        optimizer.step()
        if verbose and (i + 1) % 20 == 0:
            print('Iteration %d, content loss: %f, style loss: %f, combined loss: %f' % (
                i + 1, c_loss.item(), s_loss.item(), loss.item()))

    result = deprocess(img.data.cpu())
    result.save(output)


if __name__ == "__main__":
    # default params
    params = {
        'content_image': 'resources/slope.jpg',
        'style_image': 'resources/starry.jpg',
        'content_layer': 21,
        'content_weight': 1,
        'style_layers': (2, 7, 12, 21, 30),
        'style_weights': (9e6, 9e6, 9e6, 9e6, 9e6),
        'learning_rate': 8e-2,
        'iterations': 200,
        'verbose': True,
        'output': 'output.jpg'
    }
    style_transfer(**params)
