import torch
import torch.nn as nn
import torchaudio.transforms as T
from torchvision import transforms
from nnAudio import features
import pdb

class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_features, sample_rate=16000, window_length=250, hop_length=64, RGB=False, pretrained=False):
        super(Feature_Extraction_Layer, self).__init__()
        
        #Convert window and hop length to ms
        window_length /= 1000
        hop_length /= 1000
        
        if RGB:
            num_channels = 3
        else:
            num_channels = 1

        self.input_features = input_features

        #Return Mel Spectrogram that is 48 x 48
        self.Mel_Spectrogram = nn.Sequential(features.mel.MelSpectrogram(sample_rate,n_mels=40,win_length=int(window_length*sample_rate),
                                            hop_length=int(hop_length*sample_rate),
                                            n_fft=int(window_length*sample_rate), verbose=False), nn.ZeroPad2d((1,4,0,4)))
        
    
        #Return MFCC that is 16 x 48
        self.MFCC = nn.Sequential(features.mel.MFCC(sr=sample_rate, n_mfcc=16, 
                                        n_fft=int(window_length*sample_rate), 
                                                win_length=int(window_length*sample_rate), 
                                                hop_length=int(hop_length*sample_rate),
                                                n_mels=48, center=False, verbose=False), nn.ZeroPad2d((1,0,4,0)))

        #Return STFT that is 48 x 48
        self.STFT = nn.Sequential(features.STFT(sr=sample_rate,n_fft=int(window_length*sample_rate), 
                                        hop_length=int(hop_length*sample_rate),
                                        win_length=int(window_length*sample_rate), 
                                        output_format='Magnitude',
                                        freq_bins=48,verbose=False), nn.ZeroPad2d((1,0,0,0)))
        
        #Return GFCC that is 64 x 48
        self.GFCC = nn.Sequential(features.Gammatonegram(sr=sample_rate,
                                                hop_length=int(hop_length*sample_rate),
                                                n_fft=int(window_length*sample_rate),
                                                verbose=False,n_bins=64), nn.ZeroPad2d((1,0,0,0)))
        

        #Return CQT that is 64 x 48
        self.CQT = nn.Sequential(features.CQT(sr=sample_rate, n_bins=64, 
                                        hop_length=int(hop_length*sample_rate),
                                        verbose=False), nn.ZeroPad2d((1,0,0,0)))
        
        #Return VQT that is 64 x 48
        self.VQT = nn.Sequential(features.VQT(sr=sample_rate,hop_length=int(hop_length*sample_rate),
                                        n_bins=64,earlydownsample=False,verbose=False), nn.ZeroPad2d((1,0,0,0)))

        self.features = {'Mel_Spectrogram': self.Mel_Spectrogram, 'MFCC': self.MFCC, 'STFT': self.STFT, 'GFCC': self.GFCC, 'CQT': self.CQT, 'VQT': self.VQT}

        
    def forward(self, x):
        transformed_features = []
        for feature in self.input_features:
            transformed_features.append(self.features[feature](x))
        combined_features = torch.stack(transformed_features, dim=1)
        return combined_features