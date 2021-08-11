# ----------------------------------------
# Written by Chen Pan
# ----------------------------------------

from networks.FCCDN import *


model_dict = {
		"FCS": FCS,
		"DED": DED,
		"FCCDN": FCCDN,
}

def GenerateNet(cfg):
    return model_dict[cfg.MODEL_NAME](
            os = cfg.MODEL_OUTPUT_STRIDE,
            num_band = cfg.BAND_NUM,
            use_se = cfg.USE_SE,
        )
