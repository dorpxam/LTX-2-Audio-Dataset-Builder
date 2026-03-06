import soundfile as sf

def get_audio_info(file_path):
    info = sf.info(file_path)
    channels = "Mono" if info.channels == 1 else "Stereo"
    
    import re
    bit_match = re.search(r'(\d+)', info.subtype)
    bit_depth = f"{bit_match.group()} bits" if bit_match else "N/A"
    
    fmt = info.format.replace('WAVEX', 'WAV')
    
    return f"{fmt} | {info.samplerate} Hz | {bit_depth} | {channels}", info.duration