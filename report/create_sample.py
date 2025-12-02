import pandas as pd

# 1. Load the huge original file (1.6 million rows)
# Note: encoding='latin-1' is required for Sentiment140
print("Loading big dataset...")
df_full = pd.read_csv(
    "training.1600000.processed.noemoticon.csv", 
    encoding="latin-1", 
    header=None,
    names=["target", "id", "date", "flag", "user", "text"]
)

# 2. THE CRUCIAL STEP: Sampling with a SEED
# random_state=42 is the "Seed". It ensures we get the exact same rows every time.
print("Sampling 50,000 rows...")
df_sample = df_full.sample(n=50000, random_state=42)

# 3. Save the small file to use in your main project
df_sample.to_csv("sample_set.csv", index=False)
print("Done! Created 'sample_set.csv' with 50,000 rows.")
