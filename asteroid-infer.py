import sys
import librosa
import torch
import torch.hub


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


model = torch.hub.load(*sys.argv[1:4])
sr = int(sys.argv[4])

for f in sys.argv[5:]:
    data = librosa.core.load(f, sr=sr)[0]
    out = model.forward(normalize_tensor_wav(torch.from_numpy(data))).detach().numpy()
    librosa.output.write_wav(f.rsplit('.', 1)[0] + '_out.wav', out, sr=sr)
