import pandas as pd

# Load the huge original file (1.6 million rows)
print("Loading big dataset...")
df_full = pd.read_csv(
    "training.1600000.processed.noemoticon.csv", 
    encoding="latin-1", 
    header=None,
    names=["target", "id", "date", "flag", "user", "text"]
)

# random_state=42 is the "Seed". It ensures we get the exact same rows every time.
print("Sampling 50,000 rows...")
df_sample = df_full.sample(n=50000, random_state=42)

# Save the small file to use in the main project
df_sample.to_csv("sample_set.csv", index=False)
print("Done! Created 'sample_set.csv' with 50,000 rows.")
