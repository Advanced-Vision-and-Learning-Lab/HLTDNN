import torch.nn as nn
from nnAudio import features

class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_feature, sample_rate=16000, window_length=250, 
                 hop_length=64, RGB=False):
        super(Feature_Extraction_Layer, self).__init__()
        #Convert window and hop length to ms
        window_length /= 1000
        hop_length /= 1000
        
        if RGB:
            num_channels = 3
            MFCC_padding = nn.ZeroPad2d((3,2,16,16))
        else:
            num_channels = 1
            MFCC_padding = nn.ZeroPad2d((1,4,0,0))
        
        self.num_channels = num_channels
        self.input_feature = input_feature

        #Return Mel Spectrogram that is 48 x 48
        self.Mel_Spectrogram = nn.Sequential(features.mel.MelSpectrogram(sample_rate,n_mels=40,win_length=int(window_length*sample_rate),
                                            hop_length=int(hop_length*sample_rate),
                                            n_fft=int(window_length*sample_rate), verbose=False), 
                                            nn.ZeroPad2d((1,0,8,0)))
        
    
        #Return MFCC that is 16 x 48 (TDNN models) or 48 x 48 (CNNs)
        self.MFCC = nn.Sequential(features.mel.MFCC(sr=sample_rate, n_mfcc=16, 
                                        n_fft=int(window_length*sample_rate), 
                                                win_length=int(window_length*sample_rate), 
                                                hop_length=int(hop_length*sample_rate),
                                                n_mels=48, center=False, verbose=False), MFCC_padding)

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

        self.features = {'Mel_Spectrogram': self.Mel_Spectrogram, 
                         'MFCC': self.MFCC, 'STFT': self.STFT, 'GFCC': self.GFCC, 
                         'CQT': self.CQT, 'VQT': self.VQT}

        
    def forward(self, x):
       
        #Extract audio feature
        x = self.features[self.input_feature](x).unsqueeze(1)
        
        #Repeat channel dimension if needed (CNNs)
        x = x.repeat(1, self.num_channels,1,1)
        
        return x
    