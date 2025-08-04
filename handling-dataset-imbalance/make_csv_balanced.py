import pandas as pd
import numpy as np

CSV_PATH = 'data/driving_log.csv'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
df = pd.read_csv(CSV_PATH,names=columns)


NEAR_ZERO_THRESHOLD = 0.05
ZERO_SAMPLE_FRACTION = 0.35  


near_zero_df = df[np.abs(df['steering']) < NEAR_ZERO_THRESHOLD]
non_zero_df = df[np.abs(df['steering']) >= NEAR_ZERO_THRESHOLD]

undersampled_zero_df = near_zero_df.sample(frac=ZERO_SAMPLE_FRACTION, random_state=42)


balanced_df = pd.concat([undersampled_zero_df, non_zero_df])


balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

balanced_df.to_csv('./balanced_data/driving_log_balanced.csv', index=False)

print(f"Original samples: {len(df)}")
print(f"Balanced samples: {len(balanced_df)}")
