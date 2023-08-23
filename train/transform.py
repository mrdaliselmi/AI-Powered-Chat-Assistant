import json
import argparse

parser = argparse.ArgumentParser(
    description="Transformation my dataset to group dataset type"
)
parser.add_argument(
    "--filepath", type=str, default="train.json", help="transformation file path "
)
parser.add_argument(
    "--save_file",
    type=str,
    default="train.json",
    help="transformation saving file path ",
)
parser.add_argument(
    "--type", type=str, default="conv", help="transformation saving file path "
)

args = parser.parse_args()

if __name__ == "__main__":
    filepath = args.filepath
    data = []
    with open(filepath, "rb") as f:
        data = json.load(f)
with open(args.save_file, "w", encoding="utf-8") as f:
    print(len(data))
    for obj in data:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")
