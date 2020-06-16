import os
import sys
import librosa
import torch
import torch.hub


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


outdir = sys.argv[1]
model = torch.hub.load(*sys.argv[2:5])
sr = int(sys.argv[5])

for f in sys.argv[6:]:
    data = librosa.core.load(f, sr=sr)[0]
    out = model.forward(normalize_tensor_wav(torch.from_numpy(data))).detach().numpy()
    librosa.output.write_wav(
        os.path.join(outdir, os.path.basename(f).rsplit('.', 1)[0] + '_out.wav'), out, sr=sr)
