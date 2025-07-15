import re
import csv

input_file = "/root/autodl-tmp/Wendell/sim.txt"
output_file = "/root/autodl-tmp/Wendell/Files/Blast_result_ABC.csv"

pattern = re.compile(r"^(\S+)\s+(\d+)\s+([\d\.eE\-]+)")

rows = []
with open(input_file, "r") as f:
    for line in f:
        m = pattern.match(line)
        if m:
            rows.append(m.groups())

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Score", "E-value"])
    writer.writerows(rows)

print(f"已输出到 {output_file}")