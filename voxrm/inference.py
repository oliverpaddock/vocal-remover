from argparse import Namespace
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from lib import dataset
from lib import nets
from lib import spec_utils
from lib import utils

import pydub

class Separator(object):

    def __init__(self, model, device=None, batchsize=1, cropsize=256, postprocess=False):
        self.model = model
        self.offset = model.offset
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess

    def _postprocess(self, X_spec, mask):
        if self.postprocess:
            mask_mag = np.abs(mask)
            mask_mag = spec_utils.merge_artifacts(mask_mag)
            mask = mask_mag * np.exp(1.j * np.angle(mask))

        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        y_spec = mask * X_mag * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        # y_spec = X_spec * mask
        # v_spec = X_spec - y_spec

        return y_spec, v_spec

    def _separate(self, X_spec_pad, roi_size):
        X_dataset = []
        patches = (X_spec_pad.shape[2] - 2 * self.offset) // roi_size
        for i in range(patches):
            start = i * roi_size
            X_spec_crop = X_spec_pad[:, :, start:start + self.cropsize]
            X_dataset.append(X_spec_crop)

        X_dataset = np.asarray(X_dataset)

        self.model.eval()
        with torch.no_grad():
            mask_list = []
            # To reduce the overhead, dataloader is not used.
            for i in tqdm(range(0, patches, self.batchsize)):
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(X_batch).to(self.device)

                mask = self.model.predict_mask(torch.abs(X_batch))

                mask = mask.detach().cpu().numpy()
                mask = np.concatenate(mask, axis=2)
                mask_list.append(mask)

            mask = np.concatenate(mask_list, axis=2)

        return mask

    def separate(self, X_spec):
        n_frame = X_spec.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_spec_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= np.abs(X_spec).max()

        mask = self._separate(X_spec_pad, roi_size)
        mask = mask[:, :, :n_frame]

        y_spec, v_spec = self._postprocess(X_spec, mask)

        return y_spec, v_spec

    def separate_tta(self, X_spec):
        n_frame = X_spec.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_spec_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= X_spec_pad.max()

        mask = self._separate(X_spec_pad, roi_size)

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        X_spec_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= X_spec_pad.max()

        mask_tta = self._separate(X_spec_pad, roi_size)
        mask_tta = mask_tta[:, :, roi_size // 2:]
        mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5

        y_spec, v_spec = self._postprocess(X_spec, mask)

        return y_spec, v_spec

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'baseline.pth')

def extract(filepath:str, 
            fmt:str='wav',
            gpu:int=-1, 
            pretrained_model:str='models/baseline.pth', 
            sr:int=44100, 
            n_fft:int=2048, 
            hop_length:int=1024,
            batchsize:int=4,
            cropsize:int=256,
            output_image=False,
            tta=False,
            postprocess=False,
            output_dir:str=''
            ):
    
    args = Namespace(input=filepath, fmt=fmt, gpu=gpu, pretrained_model=pretrained_model, sr=sr, n_fft=n_fft, hop_length=hop_length, batchsize=batchsize, cropsize=cropsize, output_image=output_image, tta=tta, postprocess=postprocess, output_dir=output_dir)

    print('loading model...', end=' ')
    device = torch.device('cpu')
    if args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(args.gpu))
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
    model = nets.CascadedNet(args.n_fft, args.hop_length, 32, 128)
    model.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
    model.to(device)
    print('done')

    print('loading wave source...', end=' ')
    X, sr = librosa.load(
        args.input, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast'
    )
    basename = os.path.splitext(os.path.basename(args.input))[0]
    print('done')

    if X.ndim == 1:
        # mono to stereo
        X = np.asarray([X, X])

    print('stft of wave source...', end=' ')
    X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
    print('done')

    sp = Separator(
        model=model,
        device=device,
        batchsize=args.batchsize,
        cropsize=args.cropsize,
        postprocess=args.postprocess
    )

    if args.tta:
        y_spec, v_spec = sp.separate_tta(X_spec)
    else:
        y_spec, v_spec = sp.separate(X_spec)

    print('validating output directory...', end=' ')
    output_dir = args.output_dir
    if output_dir != "":  # modifies output_dir if theres an arg specified
        output_dir = output_dir.rstrip('/') + '/'
        os.makedirs(output_dir, exist_ok=True)
    print('done')

    print('inverse stft of instruments...', end=' ')
    wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
    print('done')
    if args.fmt.lower() == 'wav':
        sf.write('{}{}_Instruments.wav'.format(output_dir, basename), wave.T, sr)
    elif args.fmt.lower() == 'mp3':
        norm_wave = np.int16(wave * 2**15)
        channels=norm_wave.shape[0]
        song = pydub.AudioSegment(norm_wave.tobytes(), frame_rate=sr/2, sample_width=2, channels=channels)
        song.export('{}{}_Instruments.mp3'.format(output_dir, basename), format="mp3", bitrate="320k")
    else:
        print(f'invalid export type: {args.fmt}')

    print('inverse stft of vocals...', end=' ')
    wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
    print('done')
    if args.fmt.lower() == 'wav':
        sf.write('{}{}_Vocals.wav'.format(output_dir, basename), wave.T, sr)
    elif args.fmt.lower() == 'mp3':
        norm_wave = np.int16(wave * 2**15)
        channels=norm_wave.shape[0]
        song = pydub.AudioSegment(norm_wave.tobytes(), frame_rate=sr/2, sample_width=2, channels=channels)
        song.export('{}{}_Vocals.mp3'.format(output_dir, basename), format="mp3", bitrate="320k")
    else:
        print(f'invalid export type: {args.fmt}')

    if args.output_image:
        image = spec_utils.spectrogram_to_image(y_spec)
        utils.imwrite('{}{}_Instruments.jpg'.format(output_dir, basename), image)

        image = spec_utils.spectrogram_to_image(v_spec)
        utils.imwrite('{}{}_Vocals.jpg'.format(output_dir, basename), image)

if __name__ == '__main__':
    extract()