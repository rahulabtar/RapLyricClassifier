import json
from transformers import AutoTokenizer
from pathlib import Path

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

input_root = Path("cleaned_lyrics")
out_path = Path("tokenized_dataset.jsonl")

with open(out_path, "w", encoding="utf-8") as out_f:
    for artist_dir in input_root.iterdir():
        if not artist_dir.is_dir():
            continue

        for txt_file in artist_dir.glob("*.txt"):
            text = txt_file.read_text(encoding="utf-8")
            tokens = tokenizer.encode(text, add_special_tokens=True)

            record = {
                "artist": artist_dir.name,
                "text": text,
                "tokens": tokens,
                "file": txt_file.name
            }

            out_f.write(json.dumps(record) + "\n")

            print("Wrote:", txt_file)
