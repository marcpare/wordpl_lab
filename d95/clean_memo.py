import json

with open("d95/d95_third_guess_memo.json") as f:
    memo = json.load(f)

output = {}

for path, w3 in memo.items():
    w1, c1, w2, c2 = path.split(" ")
    w1 = int(w1)
    c1 = int(c1)
    w2 = int(w2)
    c2 = int(c2)
    w3 = int(w3)

    if c1 not in output:
        output[c1] = {}

    output[c1][c2] = w3

with open("d95/d953.json", "w") as f:
    json.dump(output, f)

with open("d95/d953.json", "r") as f:
    ins = json.load(f)

print(ins.keys())