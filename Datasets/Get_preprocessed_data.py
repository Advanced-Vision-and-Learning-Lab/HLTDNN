import os
import librosa
import soundfile as sf
import math

def Generate_Segments(dataset_dir, segments_dir, target_sr=16000, segment_length=3):
    '''
    dataset_dir: Directory containing DeepShip data folder
    segments_dir: Directory to save segments
    target_sr: Desired sampling rate in Hz
    segment_length: Desired segment length in seconds
    '''

    # Set the folder path containing the WAV files and subfolders
    ship_type = ['Cargo', 'Passengership', 'Tanker', 'Tug']

    for ship in ship_type:
        folder_path = '{}{}'.format(dataset_dir, ship)
        segments_path = '{}{}'.format(segments_dir, ship)

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
                    duration = len(audio_resampled)
                    segment_duration = target_sr * segment_length
                    number = math.ceil(duration / segment_duration)
                    for i in range(number):
                        start_i = int(i * segment_duration)
                        end_i = int(start_i + segment_duration)
                        if end_i > duration:
                            end_i = duration
                        output_music = audio_resampled[start_i:end_i]
                        if end_i - start_i == segment_duration:
                            segment_file_path = os.path.join(segments_path, subfolder_name,
                                                             f'{os.path.splitext(file_name)[0]}_{ship}-Segment_{i + 1}.wav')
                            os.makedirs(os.path.dirname(segment_file_path), exist_ok=True)
                            sf.write(segment_file_path, output_music, samplerate=target_sr)


def process_data(data_dir='./Datasets/DeepShip/', sample_rate=None, segment_length=None):
    segments_dir = '{}Segments/'.format(data_dir)

    # Check if the 'Segments' folder already exists
    if not os.path.exists(segments_dir):
        # If not, create the 'Segments' folder
        os.makedirs(segments_dir)
        print(f"Segments folder is creating at {segments_dir}")
        # Generate segments
        Generate_Segments(data_dir, segments_dir,
                          target_sr=sample_rate,
                          segment_length=segment_length)
    else:
        print("Segments folder already exists. Skipping segment generation.")

if __name__ == "__main__":
    process_data()

