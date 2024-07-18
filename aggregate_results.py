# python code to aggregate results for submission
# be sure to run late_fusion with intenden methods before running this

from cProfile import label
import pandas as pd
import glob
from pathlib import Path

# set result dir
result_dir = "/home/bagus/data/MuSe2024/results/prediction_muse/lf/perception"

# read all prediction files in subdir
all_files = glob.glob(f"{result_dir}/*/*_test_lf.csv", recursive=True)

# read all files
df_from_each_file = [pd.read_csv(f) for f in all_files]

# get last parent dir from each file
last_dir = [Path(f).parent.name for f in all_files]

# concate last_column ('last_dir' column) form all files
last_columns = pd.concat([df['prediction'] for df in df_from_each_file], axis=1)

# Get the column names from last_dir
column_names = last_dir

# Ensure the number of columns matches the number of names
assert len(last_columns.columns) == len(column_names), "Number of columns doesn't match number of names"

# Rename the columns
last_columns.columns = column_names

# use index from any df_from_each_file, name it 'subj_id
last_columns['subj_id'] = df_from_each_file[0]['meta_col_0']

# sort the columnt to the following order
# aggressive,arrogant,dominant,enthusiastic,friendly,leader_like,likeable,assertiv,confident,independent,risk,sincere,collaborative,kind,warm,good_natured
concat_df = last_columns[['subj_id', 'aggressive', 'arrogant', 'dominant', 'enthusiastic', 'friendly', 'leader_like', 'likeable', 'assertiv', 'confident', 'independent', 'risk','sincere', 'collaborative', 'kind', 'warm', 'good_natured']]
concat_df.to_csv("predictions.csv", index=False)
print("Result saved to predictions.csv")