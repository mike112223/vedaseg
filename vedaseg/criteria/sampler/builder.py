#from .seg_wrapper import CriterionWrapper
from vedaseg.utils import build_from_cfg

from .registry import SAMPLERS


def build_sampler(cfg):
    #criterion = CriterionWrapper(cfg)
    sampler = build_from_cfg(cfg, SAMPLERS, src='registry')
    return sampler
