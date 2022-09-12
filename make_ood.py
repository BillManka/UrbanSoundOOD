import pandas as pd
from pathlib import Path

base_path = Path('/home/wim17006/UrbanSoundOOD/UrbanSound8K')
audio_in_path = base_path / 'audio_in'
annotation_path = base_path / 'metadata/UrbanSound8K.csv'
ood_path = base_path / 'audio_ood'

df = pd.read_csv(annotation_path)
print(df.shape)

drops = []

for fold in audio_in_path.iterdir():
    for wav in fold.iterdir():
        filename = wav.name
        class_id = df.loc[df['slice_file_name']==wav.name, 'classID'].item()
	
        if class_id == 6:
            wav.rename(ood_path / wav.name)
            row_index = df.loc[df['slice_file_name'] == wav.name, :].index.item()
            print(row_index)
            drops.append(row_index)


print(drops)
df = df.drop(labels=drops, axis=0)
print(df.shape)

df.to_csv(base_path / 'metadata/reduced.csv', index=False) 


