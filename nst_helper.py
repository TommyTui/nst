import torch
import torchvision.transforms as T

VGG19_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
VGG19_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)


def preprocess(img):
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=VGG19_MEAN.tolist(), std=VGG19_STD.tolist()),
            T.Lambda(lambda x: x[None]),
        ]
    )
    return transform(img)


def deprocess(img):
    transform = T.Compose(
        [
            T.Lambda(lambda x: x[0]),
            T.Normalize(mean=[0, 0, 0], std=(1.0 / VGG19_STD).tolist()),
            T.Normalize(mean=(-VGG19_MEAN).tolist(), std=[1, 1, 1]),
            T.Lambda(rescale),
            T.ToPILImage(),
        ]
    )
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def extract_features(imgs, cnn):
    features = []
    prev_feat = imgs
    for _, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features
