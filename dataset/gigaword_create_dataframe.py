import pandas as pd

test_src_data_path = "/mnt/hdd/karmpatel/naman/demo/DLNLP_Project_Ckpts/org_data/test.src.txt"
test_tgt_data_path = "/mnt/hdd/karmpatel/naman/demo/DLNLP_Project_Ckpts/org_data/test.tgt.txt"

# Open and read the file line by line
test_src_line_cnt = 0
test_tgt_line_cnt = 0
src = []
tgt = []

with open(test_src_data_path, 'r', encoding='utf-8') as file:
    for line in file:
        src.append(line.strip().lower())

with open(test_tgt_data_path, 'r', encoding='utf-8') as file:
    for line in file:
        tgt.append(line.strip().lower())

# Check if the lists are of equal length
if len(src) != len(tgt):
    print("Error: Source and Target lists are of different lengths!")
else:
    # Convert to DataFrame
    data = pd.DataFrame({
        "document": src,
        "summary": tgt
    })

    # Display the first few rows of the DataFrame
    print(data)

    # Save to a CSV file (optional)
    data.to_csv("../dataset/gigaword/test.csv", index=False)

