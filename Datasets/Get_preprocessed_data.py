# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 11:20:44 2023
Generate segments for DeepShip Dataset
@author: JarinRitu 
"""
#----------------------Structure of the dataset------------------

'''Deepship/
    ├── Cargo/
    │   ├── Cargo1/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   ├── Cargo2/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   └── ...
    ├── Passenger/
    │   ├── Passenger1/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   ├── Passenger2/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   └── ...
    ├── Tanker/
    │   ├── Tanker1/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   ├── Tanker2/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   └── ...
    ├── Tug/
    │   ├── Tug1/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   ├── Tug2/
    │   │   ├── audio1.wav
    │   │   ├── audio2.wav
    │   │   └── ...
    │   └── ...
    └── ...

'''


import os
import librosa
import soundfile as sf
import math

# Set the folder path containing the WAV files and subfolders
ship_type = ['Cargo', 'Passengership', 'Tanker', 'Tug']

# Set the target sampling rate and segment length
target_sr = int(input("Enter the target sampling rate (hz): "))
segment_length = int(input("Enter the target segment length (seconds): "))



for ship in ship_type:
    folder_path = './Datasets/DeepShip/{}'.format(ship)
    segments_path = './Datasets/DeepShip/Segments/{}'.format(ship)
    
    # Loop over all subfolders in the parent folder
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)

        # Only process subfolders
        if not os.path.isdir(subfolder_path):
            continue
    
        # Loop over all WAV files in the subfolder
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.wav'):
                # Load the audio signal
                file_path = os.path.join(subfolder_path, file_name)
                audio, sr = librosa.load(file_path, sr=None)
    
                # Resample to the target sampling rate
                audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
                # Divide the resampled audio into segments and save them to a folder
                start = 0
                duration = len(audio_resampled)
                segment_duration = target_sr * segment_length
                number = math.ceil(duration / segment_duration)
                for i in range(number):
                  start_i = i * segment_duration
                  end_i = start_i + segment_duration
                  if end_i > duration:
                    end_i = duration
                  output_music = audio_resampled[start_i:end_i]
                  if end_i - start_i == segment_duration:
                    segment_file_path = os.path.join(segments_path, subfolder_name, f'{os.path.splitext(file_name)[0]}_{ship}-Segment_{i+1}.wav')
                    os.makedirs(os.path.dirname(segment_file_path), exist_ok=True)
                    sf.write(segment_file_path, output_music, samplerate=target_sr)
