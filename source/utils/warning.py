import warnings
import logging
import os

def silence_warnings():
    # 1. Ignorer les warnings Python/User (pyloudnorm, torchcodec, bitsandbytes)
    warnings.filterwarnings("ignore", message="The 'bits_per_sample' parameter is not directly supported")
    warnings.filterwarnings("ignore", message="Possible clipped samples in output")
    warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16")
    
    # 2. Réduire le verbiage de Transformers
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()

    # 3. Optionnel : Désactiver les logs de version de CUDA/FlashAttn si besoin
    os.environ["TOKENIZERS_PARALLELISM"] = "false"