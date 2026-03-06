import torch
import re
import os
from qwen_asr import Qwen3ASRModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging as transformers_logging

from source.utils.logger import get_logger
from source.utils.translator import T
from source.utils.console import Console
from source.common.config import EngineConfig

transformers_logging.set_verbosity_error()

logger = get_logger("Engine")
console = Console()

class InferenceEngine:
    def __init__(self, config: EngineConfig = None):
        self.config = config if config else EngineConfig()
        
        self.device = self.config.device
        self.asr_id = self.config.asr_model_id
        self.llm_id = self.config.llm_id
        
        self.asr_model = None
        self.llm_model = None
        self.llm_tokenizer = None

    def transcribe(self, audio_path: str, language: str = "English"):
        try:
            res = self.asr_model.transcribe(audio=audio_path, language=language)
            
            if isinstance(res, list) and len(res) > 0:
                res_obj = res[0]
            else:
                res_obj = res
            
            if hasattr(res_obj, 'text'):
                return res_obj.text.strip()
            
            return str(res_obj).strip()
            
        except Exception as e:
            logger.error(T.translate("engine_error_transcription", error=e))
            return None

    def analyze_style(self, text: str):
        prompt = self.config.llm_prompt.format(text=text)
        
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs, 
                max_new_tokens=self.config.llm_max_tokens, 
                temperature=self.config.llm_temp
            )
        
        response = self.llm_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        match = re.search(r'\d+', response.strip())
        if match:
            score = int(match.group())
            score = min(score, 10)
        else:
            score = 5

        logger.debug(T.translate("engine_llm_debug_score", resp=response.strip(), score=score))
        return score

    def free_vram(self, target: str = "all"):
        if target in ["asr", "all"]:
            if self.asr_model is not None:
                del self.asr_model
                self.asr_model = None
                logger.debug(T.translate("engine_vram_asr_removed"))
                
        if target in ["llm", "all"]:
            if self.llm_model is not None:
                del self.llm_model
                self.llm_model = None
                self.llm_tokenizer = None
                logger.debug(T.translate("engine_vram_llm_removed"))
        
        torch.cuda.empty_cache()
        
    def _load_asr(self):
        if self.asr_model is None:
            logger.info(T.translate("engine_loading_asr", model=self.asr_id))
            try:
                console.separator()
                self.asr_model = Qwen3ASRModel.from_pretrained(
                    self.asr_id, 
                    dtype=torch.bfloat16 if not self.config.use_8bit else None,
                    device_map=self.device,
                    attn_implementation="flash_attention_2" if self.config.use_flash_attn_2 else None,
                    low_cpu_mem_usage=self.config.use_low_cpu_mem_usage,
                    pad_token_id=151645
                )
                console.separator()
            except Exception as e:
                logger.error(T.translate("engine_load_error", model="ASR", error=e))
                raise

    def ensure_asr_loaded(self):
        self._load_asr()

    def _load_llm(self):
        if self.llm_model is None:
            bits_label = "8-bit" if self.config.use_8bit else "BF16"
            logger.info(T.translate("engine_loading_llm", model=self.llm_id, bits=bits_label))
            try:
                console.separator()
                q_config = BitsAndBytesConfig(load_in_8bit=True) if self.config.use_8bit else None
                self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_id)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_id,
                    device_map=self.device,
                    quantization_config=q_config,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2" if self.config.use_flash_attn_2 else None,
                    low_cpu_mem_usage=self.config.use_low_cpu_mem_usage,
                )
                console.separator()
            except Exception as e:
                logger.error(T.translate("engine_load_error", model="LLM", error=e))
                raise

    def ensure_llm_loaded(self):
        self._load_llm()