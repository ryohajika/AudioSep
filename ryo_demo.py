# download https://audio-agi.github.io/Separate-Anything-You-Describe/demos/exp31_water drops_mixture.wav and place it to the same directory
# also download https://huggingface.co/spaces/badayvedat/AudioSep/resolve/main/checkpoint/audiosep_base_4M_steps.ckpt and put under ./checkpoint 

import torch

# the following two lines are the key
import numpy
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar, numpy.dtype, numpy.dtypes.Float64DType])

from pipeline import build_audiosep, separate_audio

# for the fellow mac users...
device = torch.device('mps' if torch.mps.is_available() else 'cpu')

model = build_audiosep(
      config_yaml='config/audiosep_base.yaml',
      checkpoint_path='checkpoint/audiosep_base_4M_steps.ckpt',
      device=device)

audio_file = 'exp31_water drops_mixture.wav'
text = 'water drops'
output_file='separated_audio.wav'

# AudioSep processes the audio at 32 kHz sampling rate
separate_audio(model, audio_file, text, output_file, device)