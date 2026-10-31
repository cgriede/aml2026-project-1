from pathlib import Path
import hashlib
file_1 = Path(r"/home/cedric/code/mini-projects/eth/AML-2025-Project-1/data/X_train.csv")
file_2 = Path(r"/home/cedric/code/mini-projects/eth/aml26-1/eth-aml-2026-project-1/X_train.csv")

file_1_hash = hashlib.sha256(file_1.read_bytes()).hexdigest()
file_2_hash = hashlib.sha256(file_2.read_bytes()).hexdigest()

print(file_1_hash)
print(file_2_hash)

if file_1_hash == file_2_hash:
    print("The files are the same")
else:
    print("The files are different")
