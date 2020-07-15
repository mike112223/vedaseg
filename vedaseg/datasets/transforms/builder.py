from vedaseg.utils import build_from_cfg

from .registry import TRANSFORMS

from .transforms import Compose


def build_transform(cfg, bicfg):
    tfs = []
    for icfg in cfg:
        tf = build_from_cfg(icfg, TRANSFORMS)
        tfs.append(tf)

    bitfs = []
    if bicfg is not None:
        for biicfg in bicfg:
            bitf = build_from_cfg(biicfg, TRANSFORMS)
            bitfs.append(bitf)

    aug = Compose(tfs, bitfs)

    return aug
