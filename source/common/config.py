import yaml
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

@dataclass
class EngineConfig:
    device: str = "cuda:0"
    use_8bit: bool = False
    use_flash_attn_2: bool = True
    use_low_cpu_mem_usage: bool = False
    
    asr_model_id: str = ""
    llm_id: str = ""
    llm_temp: float = 0.1
    llm_max_tokens: int = 10
    llm_prompt: str = ""

@dataclass
class PipelineConfig:
    audio_source: str
    output_dir: str
    settings: SimpleNamespace = None
    config: SimpleNamespace = None
    engine: EngineConfig = field(default_factory=EngineConfig)
    
    @property
    def segmentation(self): return self.config.segmentation
    @property
    def scoring(self): return self.config.scoring
    @property
    def analyze(self): return self.config.analyze
    @property
    def caption(self): return self.config.caption

def dict_to_sns(data):
    return json.loads(json.dumps(data), object_hook=lambda d: SimpleNamespace(**d))

def load_all_configs(audio_source: str, output_dir: str, user_config_path: str = None) -> PipelineConfig:
    base_config_path = Path(__file__).parent.parent / "config"
    settings_file = base_config_path / "settings.yaml"
    
    if user_config_path and os.path.exists(user_config_path):
        config_file = Path(user_config_path)
    else:
        config_file = base_config_path / "config.yaml"

    with open(settings_file, 'r', encoding='utf-8') as f:
        s_data = yaml.safe_load(f)
    with open(config_file, 'r', encoding='utf-8') as f:
        c_data = yaml.safe_load(f)

    h = c_data.get('hardware', {})
    m = s_data.get('models', {})
    
    engine_cfg = EngineConfig(
        device=h.get('device', "cuda:0"),
        use_8bit=h.get('use_8bit', False),
        use_flash_attn_2=h.get('use_flash_attn_2', True),
        use_low_cpu_mem_usage=h.get('use_low_cpu_mem_usage', False),
        asr_model_id=m.get('asr', {}).get('id', ""),
        llm_id=m.get('llm', {}).get('id', ""),
        llm_temp=m.get('llm', {}).get('temperature', 0.1),
        llm_max_tokens=m.get('llm', {}).get('max_new_tokens', 10),
        llm_prompt=m.get('llm', {}).get('style_prompt', "")
    )

    settings_obj = dict_to_sns(s_data)
    user_config_obj = dict_to_sns(c_data)

    return PipelineConfig(
        audio_source=audio_source,
        output_dir=output_dir,
        settings=settings_obj,
        config=user_config_obj,
        engine=engine_cfg
    )