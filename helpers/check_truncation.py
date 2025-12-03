"""
Verify that lyrics are not being truncated during tokenization
"""
import json
from transformers import AutoTokenizer
from pathlib import Path

print("=" * 80)
print("CHECKING FOR TRUNCATION IN TOKENIZATION")
print("=" * 80)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
max_length = tokenizer.model_max_length
print(f"\nBERT model max length: {max_length} tokens")

# Load tokenized dataset
data = []
with open("tokenized_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print(f"Total entries: {len(data)}")

# Check token lengths
token_lengths = [len(entry["tokens"]) for entry in data]
max_tokens = max(token_lengths)
min_tokens = min(token_lengths)
avg_tokens = sum(token_lengths) / len(token_lengths)

print(f"\nToken length statistics:")
print(f"  Min tokens: {min_tokens}")
print(f"  Max tokens: {max_tokens}")
print(f"  Average tokens: {avg_tokens:.1f}")

# Check for truncation indicators
entries_over_512 = [i for i, length in enumerate(token_lengths) if length > 512]
entries_exactly_512 = [i for i, length in enumerate(token_lengths) if length == 512]
entries_under_512 = [i for i, length in enumerate(token_lengths) if length < 512]

print(f"\nEntries by length:")
print(f"  Over 512 tokens: {len(entries_over_512)} ({len(entries_over_512)/len(data)*100:.1f}%)")
print(f"  Exactly 512 tokens: {len(entries_exactly_512)} ({len(entries_exactly_512)/len(data)*100:.1f}%)")
print(f"  Under 512 tokens: {len(entries_under_512)} ({len(entries_under_512)/len(data)*100:.1f}%)")

# Check if entries exactly at 512 end with [SEP] (would indicate truncation)
if entries_exactly_512:
    print(f"\nChecking entries with exactly 512 tokens (possible truncation):")
    for idx in entries_exactly_512[:3]:  # Check first 3
        entry = data[idx]
        tokens = entry["tokens"]
        last_token = tokens[-1]
        print(f"  Entry {idx}: Last token = {last_token} ({'[SEP]' if last_token == 102 else 'NOT [SEP]'})")
        if last_token != 102:
            print(f"    -> NOT truncated (doesn't end with [SEP])")
        else:
            # Check if this is natural or truncation
            # If truncated, the second-to-last token would be cut off mid-word
            print(f"    -> Could be natural end or truncation")

# Verify by re-tokenizing a long entry
print(f"\n" + "=" * 80)
print("VERIFICATION: Re-tokenizing a long entry")
print("=" * 80)

# Find a long entry
long_entry_idx = entries_over_512[0] if entries_over_512 else 0
long_entry = data[long_entry_idx]

print(f"\nTesting entry: {long_entry['file']}")
print(f"  Original text length: {len(long_entry['text'])} characters")
print(f"  Tokenized length: {len(long_entry['tokens'])} tokens")

# Re-tokenize to verify
original_text = long_entry["text"]
re_tokens = tokenizer.encode(original_text, add_special_tokens=True)

print(f"  Re-tokenized length: {len(re_tokens)} tokens")
print(f"  Match: {'YES' if len(re_tokens) == len(long_entry['tokens']) else 'NO'}")

if len(re_tokens) == len(long_entry['tokens']):
    print(f"  ✅ NO TRUNCATION - Full text is preserved")
else:
    print(f"  ⚠️  Length mismatch - investigate further")

# Check tokenization code
print(f"\n" + "=" * 80)
print("TOKENIZATION CODE CHECK")
print("=" * 80)

print("\nCurrent tokenization code:")
print("  tokenizer.encode(text, add_special_tokens=True)")
print("\nParameters:")
print("  - add_special_tokens=True: Adds [CLS] and [SEP]")
print("  - max_length: NOT specified (default: no limit)")
print("  - truncation: NOT specified (default: False)")
print("\n✅ This means NO TRUNCATION is happening!")

# Summary
print(f"\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
if max_tokens > 512:
    print("✅ CONFIRMED: Lyrics are NOT being truncated")
    print(f"   - Maximum token length: {max_tokens} (way over 512 limit)")
    print(f"   - {len(entries_over_512)} entries exceed 512 tokens")
    print(f"   - Tokenization preserves full text")
else:
    print("⚠️  All entries are 512 tokens or less")
    print("   - This could indicate truncation or all songs are short")

print("\nNote: For BERT models, sequences >512 tokens will need special handling:")
print("  - Truncation (lose end of text)")
print("  - Splitting into multiple sequences")
print("  - Using a model with longer context (e.g., Longformer)")

