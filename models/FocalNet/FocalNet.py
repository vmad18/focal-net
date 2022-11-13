from models.Layers.FocalNet import FocalBlock
from utils.Layers import PatchEmbedding
from utils.consts import *


class FocalNet(Module):

    def __init__(self, dims: int = 96, depths=[2, 2, 3, 2], levels: int = 3, k_s: int = 3, scale_r: int = 4, dp: float = 0., layer_scale: float = 1e-4):
        super().__init__()

        self.model = nn.ModuleList([])

        self.inp_embed = PatchEmbedding(3, dims)  # does not require positional embedding as focal modulation is invariant to translation

        for i, depth in enumerate(depths):
            self.model.append(FocalBlock(dims * 2**i, dims * 2**(i+1), depth, levels, k_s, scale_r, dp, layer_scale, i == len(depths)-1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.inp_embed(x)
        for block in self.model:
            x = block(x)

        return adaptpool2d(x, (1, 1)).flatten(1)


if __name__ == "__main__":
    tnsr = torch.randn(1, 3, 224, 224)
    fn = FocalNet()
    print(fn(tnsr).shape)
