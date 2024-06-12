# segment Muse 2024 Humor audio data using pydub

import os
from pydub import AudioSegment
import glob

# list audio files given a directory
input_dir = ('/data/MuSe2024/c2_muse_humor/raw_data/audio/')
audio_files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)

for audio_file in audio_files:
    # segment into 2 seconds with 1 second overlap
    audio = AudioSegment.from_file(audio_file)

    segment_length = 2000  # 2 seconds in milliseconds
    overlap = 1000  # 1 second overlap in milliseconds
    segments = []

    for i in range(0, len(audio), segment_length - overlap):
        segment = audio[i : i + segment_length]
        segments.append(segment)

    # save segments in the same parent directory name 
    output_dir = '/data/MuSe2024/c2_muse_humor/segmented_data/audio/'
    # os.makedirs(output_dir, exist_ok=True)
    for i, segment in enumerate(segments):
        # get parent directory
        parent_dir = os.path.basename(os.path.dirname(audio_file))
        # make output directory if not exist
        os.makedirs(os.path.join(output_dir, parent_dir), exist_ok=True)
        segment.export(os.path.join(output_dir, parent_dir, f"{os.path.basename(audio_file)[:-4]}_{i}.wav"), format='wav')
        print(f"Segmented {audio_file} into {len(segments)} segments")
    
    print("DONE.")
    # segment Muse 2024 Humor audio data using pydub
        
