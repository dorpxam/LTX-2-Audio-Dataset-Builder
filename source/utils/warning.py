import warnings
import logging
import os

def silence_warnings():
    warnings.filterwarnings("ignore", message="The 'bits_per_sample' parameter is not directly supported")
    warnings.filterwarnings("ignore", message="Possible clipped samples in output")
    warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16")
    
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"