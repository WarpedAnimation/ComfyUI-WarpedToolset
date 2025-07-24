import torch
import tempfile, time
import gradio as gr
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple
from diao.dia.model import Dia
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')
print('Loading model...')
try:
    model = Dia.from_pretrained('callgg/dia-f16', compute_dtype='float16',
        device=device)
except Exception as e:
    print(f'Error loading Nari model: {e}')
    raise
def run_inference(text_input, audio_prompt_input, max_new_tokens, cfg_scale,
    temperature, top_p, cfg_filter_top_k, speed_factor):
    """
    Runs inference using the globally loaded model and provided inputs.
    """
    global model, device
    if not text_input or text_input.isspace():
        raise gr.Error('Text input cannot be empty.')
    temp_txt_file_path = None
    temp_audio_prompt_path = None
    output_audio = 44100, np.zeros(1, dtype=np.float32)
    try:
        prompt_path_for_generate = None
        if audio_prompt_input is not None:
            sr, audio_data = audio_prompt_input
            if audio_data is None or audio_data.size == 0 or audio_data.max(
                ) == 0:
                gr.Warning(
                    'Audio prompt seems empty or silent, ignoring prompt.')
            else:
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav',
                    delete=False) as f_audio:
                    temp_audio_prompt_path = f_audio.name
                    if np.issubdtype(audio_data.dtype, np.integer):
                        max_val = np.iinfo(audio_data.dtype).max
                        audio_data = audio_data.astype(np.float32) / max_val
                    elif not np.issubdtype(audio_data.dtype, np.floating):
                        gr.Warning(
                            f'Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion.'
                            )
                        try:
                            audio_data = audio_data.astype(np.float32)
                        except Exception as conv_e:
                            raise gr.Error(
                                f'Failed to convert audio prompt to float32: {conv_e}'
                                )
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 2:
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] == 2:
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            gr.Warning(
                                f'Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis.'
                                )
                            audio_data = audio_data[0] if audio_data.shape[0
                                ] < audio_data.shape[1] else audio_data[:, 0]
                        audio_data = np.ascontiguousarray(audio_data)
                    try:
                        sf.write(temp_audio_prompt_path, audio_data, sr,
                            subtype='FLOAT')
                        prompt_path_for_generate = temp_audio_prompt_path
                        print(
                            f'Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})'
                            )
                    except Exception as write_e:
                        print(f'Error writing temporary audio file: {write_e}')
                        raise gr.Error(
                            f'Failed to save audio prompt: {write_e}')
        start_time = time.time()
        with torch.inference_mode():
            output_audio_np = model.generate(text_input, max_tokens=
                max_new_tokens, cfg_scale=cfg_scale, temperature=
                temperature, top_p=top_p, cfg_filter_top_k=cfg_filter_top_k,
                use_torch_compile=False, audio_prompt=prompt_path_for_generate)
        end_time = time.time()
        print(f'Generation finished in {end_time - start_time:.2f} seconds.')
        if output_audio_np is not None:
            output_sr = 44100
            original_len = len(output_audio_np)
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(original_len / speed_factor)
            if target_len != original_len and target_len > 0:
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                resampled_audio_np = np.interp(x_resampled, x_original,
                    output_audio_np)
                output_audio = output_sr, resampled_audio_np.astype(np.float32)
                print(
                    f'Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed.'
                    )
            else:
                output_audio = output_sr, output_audio_np
                print(
                    f'Skipping audio speed adjustment (factor: {speed_factor:.2f}).'
                    )
            print(
                f'Audio conversion successful. Final shape: {output_audio[1].shape}, Sample Rate: {output_sr}'
                )
            if output_audio[1].dtype == np.float32 or output_audio[1
                ].dtype == np.float64:
                audio_for_gradio = np.clip(output_audio[1], -1.0, 1.0)
                audio_for_gradio = (audio_for_gradio * 32767).astype(np.int16)
                output_audio = output_sr, audio_for_gradio
                print('Converted audio to int16 for Gradio output.')
        else:
            print('\nGeneration finished, but no valid tokens were produced.')
            gr.Warning('Generation produced no output.')
    except Exception as e:
        print(f'Error during inference: {e}')
        import traceback
        traceback.print_exc()
        raise gr.Error(f'Inference failed: {e}')
    finally:
        if temp_txt_file_path and Path(temp_txt_file_path).exists():
            try:
                Path(temp_txt_file_path).unlink()
                print(f'Deleted temporary text file: {temp_txt_file_path}')
            except OSError as e:
                print(
                    f'Warning: Error deleting temporary text file {temp_txt_file_path}: {e}'
                    )
        if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
            try:
                Path(temp_audio_prompt_path).unlink()
                print(
                    f'Deleted temporary audio prompt file: {temp_audio_prompt_path}'
                    )
            except OSError as e:
                print(
                    f'Warning: Error deleting temporary audio prompt file {temp_audio_prompt_path}: {e}'
                    )
    return output_audio
css = """
"""
default_text = """[S1] This is an open weights text to dialogue model. 
[S2] You get full control over scripts and voices. 
[S1] Wow. Amazing. (laughs) 
[S2] Try it now on Git hub or Hugging Face."""
block = gr.Blocks(title='gguf', css=css).queue()
with block:
    gr.Markdown('## 🎤 Text-to-Speech Synthesis')
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            text_input = gr.Textbox(label='Input Text', placeholder=
                'Enter text here...', value=default_text, lines=5)
            audio_prompt_input = gr.Audio(label='Audio Prompt (Optional)',
                show_label=True, sources=['upload', 'microphone'], type='numpy'
                )
            with gr.Accordion('Generation Parameters', open=False):
                max_new_tokens = gr.Slider(label=
                    'Max New Tokens (Audio Length)', minimum=860, maximum=
                    3072, value=model.config.data.audio_length, step=50,
                    info=
                    'Controls the maximum length of the generated audio (more tokens = longer audio).'
                    )
                cfg_scale = gr.Slider(label='CFG Scale (Guidance Strength)',
                    minimum=1.0, maximum=5.0, value=3.0, step=0.1, info=
                    'Higher values increase adherence to the text prompt.')
                temperature = gr.Slider(label='Temperature (Randomness)',
                    minimum=1.0, maximum=1.5, value=1.3, step=0.05, info=
                    'Lower values make the output more deterministic, higher values increase randomness.'
                    )
                top_p = gr.Slider(label='Top P (Nucleus Sampling)', minimum
                    =0.8, maximum=1.0, value=0.95, step=0.01, info=
                    'Filters vocabulary to the most likely tokens cumulatively reaching probability P.'
                    )
                cfg_filter_top_k = gr.Slider(label='CFG Filter Top K',
                    minimum=15, maximum=50, value=30, step=1, info=
                    'Top k filter for CFG guidance.')
                speed_factor_slider = gr.Slider(label='Speed Factor',
                    minimum=0.8, maximum=1.0, value=0.94, step=0.02, info=
                    'Adjusts the speed of the generated audio (1.0 = original speed).'
                    )
            run_button = gr.Button('Generate Audio', variant='primary')
        with gr.Column(scale=1):
            audio_output = gr.Audio(label='Generated Audio', type='numpy',
                autoplay=False)
    run_button.click(fn=run_inference, inputs=[text_input,
        audio_prompt_input, max_new_tokens, cfg_scale, temperature, top_p,
        cfg_filter_top_k, speed_factor_slider], outputs=[audio_output],
        api_name='generate_audio')
block.launch()