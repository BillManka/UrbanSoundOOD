import pandas as pd
from pathlib import Path

base_path = Path('/home/wim17006/UrbanSoundOOD/UrbanSound8K')
audio_in_path = base_path / 'audio_in'
annotation_path = base_path / 'metadata/UrbanSound8K.csv'
ood_path = base_path / 'audio_ood'

df = pd.read_csv(annotation_path)
print(df.shape)

drop_indices = []
gunshot_rows = []

for fold in audio_in_path.iterdir():
    for wav in fold.iterdir():
        filename = wav.name
        
        class_id = df.loc[df['slice_file_name']==wav.name, 'classID'].item()

        if class_id == 6:
            wav.rename(ood_path / wav.name)
            row = df.loc[df['slice_file_name'] == wav.name, :].iloc[0, :]
            gunshot_rows.append(row.ravel())
            row_index = df.loc[df['slice_file_name'] == wav.name, :].index.item()
            print(row_index)
            drop_indices.append(row_index)


df = df.drop(labels=drop_indices, axis=0)
ood_df = pd.DataFrame(gunshot_rows)

print(df.shape)

df.to_csv(base_path / 'metadata/in_annotations.csv', index=False) 
ood_df.to_csv(base_path / 'metadata/out_annotations.csv', index=False) 


