#!/bin/python3
import json

def tsv_from_json(json_file, tsv_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    columns = dict()
    num_rows = 0

    # gather all of the data and organize it into the rows and columns I will want
    for filepath in data:
        row = data[filepath]
        for key in row:
            if type(row[key]) is dict:
                for k2 in row[key]:
                    col = (key, k2)
                    if col not in columns:
                        columns[col] = [''] * num_rows
                    columns[col].extend([''] * (num_rows - len(columns[col])))
                    columns[col].append(f"{row[key][k2]}")
            else:
                col = key
                if col not in columns:
                    columns[col] = [''] * num_rows
                columns[col].append(f"{row[key]}")
        num_rows += 1

    # render the data to tsv

    with open(tsv_file, "w") as f:
        # write the header columns
        line = ''
        for col in columns:
            if type(col) is tuple:
                line += f"\t{col[0]}"
            else:
                line += f"\t{col}"
        f.write(line[1:] + "\n")

        # write the sub-header columns
        line = ''
        for col in columns:
            if type(col) is tuple:
                line += f"\t{col[1]}"
            else:
                line += f"\t"
        f.write(line[1:] + "\n")

        # write the data
        for i in range(num_rows):
            line = ''
            for col in columns:
                try:
                    line += f"\t{columns[col][i]}"
                except IndexError:
                    line += "\t"
            f.write(line[1:] + "\n")


if __name__ == "__main__":
    from sys import argv
    try:
        source = argv[1]
        dest = argv[2]
    except:
        print("Usage: python3 make_tsv.py <source.json> <dest.tsv>")
        exit(0)
    tsv_from_json(source, dest)
