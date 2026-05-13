import pickle
import gzip
import os

# ─── STEP 1: Compress your pickle file ────────────────────────────────────────

input_file  = "model.pkl"          # ← change to your pickle file name
output_file = "model.pkl.gz"       # compressed output

with open(input_file, "rb") as f_in:
    data = pickle.load(f_in)

with gzip.open(output_file, "wb", compresslevel=9) as f_out:   # 9 = max compression
    pickle.dump(data, f_out)

original_mb    = os.path.getsize(input_file)  / (1024 * 1024)
compressed_mb  = os.path.getsize(output_file) / (1024 * 1024)
reduction      = (1 - compressed_mb / original_mb) * 100

print(f"Original:   {original_mb:.1f} MB")
print(f"Compressed: {compressed_mb:.1f} MB")
print(f"Reduced by: {reduction:.1f}%")


# ─── STEP 2: Load the compressed model in your Streamlit app ──────────────────
# Replace your existing pickle.load() with this:

# import gzip, pickle
# with gzip.open("your_model.pkl.gz", "rb") as f:
#     model = pickle.load(f)