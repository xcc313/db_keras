# -*- coding: utf-8 -*-
import yaml
import os
import numpy as np
import pprint

def build_cfg(path):
    print('Loading config from ', path)
    if not os.path.exists(path):
        raise KeyError('%s does not exist ...' % path)

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f.read())
        cfg.update(cfg['global'])
        pprint.pprint(cfg)

    return cfg
