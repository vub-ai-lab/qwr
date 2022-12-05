import torch
import torchvision
import numpy as np
import gym

class FlattenExtractor(torch.nn.Module):
    """ Extractor for simple Box observation spaces: the observations are simply flattened.
    """

    def __init__(self, observation_space, args):
        super().__init__()

        self.flatten = torch.nn.Flatten()

        if args.augment:
            self.aug = lambda x: x * (1.0 + 0.1 * torch.rand_like(x))
        else:
            self.aug = lambda x: x

        self.output_len = int(np.prod(np.array(observation_space.shape)))

    def forward(self, x):
        x = self.flatten(x)
        x = self.aug(x)

        return x

class CnnExtractor(torch.nn.Module):
    """ Extractor for image observations (3D uint8 Box spaces with 1, 3 or 4 channels)
    """

    def __init__(self, observation_space, args):
        super().__init__()

        if observation_space.shape[0] in [1, 3, 4]:
            self.is_channels_last = False
            c, h, w = observation_space.shape
        else:
            # Channels last
            self.is_channels_last = True
            h, w, c = observation_space.shape

        self.is_uint8 = (observation_space.dtype.name == 'uint8')

        # We assume CxHxW images (channels first)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.cnn = to_channels_last(self.cnn)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros((1, c, h, w))).shape[1]

        if args.augment:
            self.aug = torchvision.transforms.RandomCrop(h, padding=4, pad_if_needed=True, padding_mode='edge')
        else:
            self.aug = lambda x: x

        self.output_len = 256
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, self.output_len),
            torch.nn.ReLU()
        )

    def forward(self, observations):
        # Reorder channels in observations if needed
        if self.is_channels_last:
            # Go from NHWC to NCHW
            observations = torch.transpose(observations, 1, 3)

        if self.is_uint8:
            observations = observations.float() / 255.0

        x = self.aug(observations)
        x = to_channels_last(x)
        x = self.cnn(x)

        return self.linear(x)

class DictExtractor(torch.nn.Module):
    """ Extractor for Dict observation spaces. Recursion is allowed: a Dict can
        contain a Dict for some of its key, etc.
    """

    def __init__(self, observation_space, args):
        super().__init__()

        extractors = {}
        self.output_len = 0

        for key, subspace in observation_space.spaces.items():
            extractor = make_extractor(subspace, args)

            self.output_len += extractor.output_len
            extractors[key] = extractor

        self.extractors = torch.nn.ModuleDict(extractors)

    def forward(self, x):
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded = extractor(x[key])
            encoded_tensor_list.append(encoded)

        return torch.cat(encoded_tensor_list, dim=1)

def make_extractor(space, args):
    """ Return an Extractor for a given observation space
    """
    if isinstance(space, gym.spaces.Box) and len(space.shape) == 3:
        return CnnExtractor(space, args)
    elif isinstance(space, gym.spaces.Dict):
        return DictExtractor(space, args)
    else:
        return FlattenExtractor(space, args)

def to_channels_last(x):
    """ GPU and CPU optimization: convert images to the channels last memory representation.
        The shape does not change and still remains (N, C, H, W).
    """
    return x.to(memory_format=torch.channels_last)
