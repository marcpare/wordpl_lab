import sys

if __name__ == "__main__":
    filename = sys.argv[1]
    output = [0] * (3**5)
    with open(filename, "r") as f:
        for line in f:
            c1, w2 = line.split()
            output[int(c1)] = int(w2)
    print(output)

