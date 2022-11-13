from utils.consts import *
from models.FocalNet.FocalNet import FocalNet


def main() -> null:

    tnsr = torch.randn((1, 3, 224, 224))
    focalnet = FocalNet()
    print(focalnet(tnsr).shape)


if __name__ == "__main__":
    main()
