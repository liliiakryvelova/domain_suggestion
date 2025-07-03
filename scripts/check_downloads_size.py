import os

downloads_path = os.path.expanduser("~/Downloads")
total_size = 0

for dirpath, dirnames, filenames in os.walk(downloads_path):
    for f in filenames:
        fp = os.path.join(dirpath, f)
        if os.path.isfile(fp):
            total_size += os.path.getsize(fp)

# Convert to MB
print(f"Downloads folder size: {total_size / (1024 * 1024):.2f} MB")
