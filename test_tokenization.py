import json
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Read a few entries from the tokenized dataset
with open("tokenized_dataset.jsonl", "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f.readlines()[:5]]

print("=" * 80)
print("TOKENIZATION VERIFICATION")
print("=" * 80)

for i, entry in enumerate(entries, 1):
    print(f"\n--- Entry {i}: {entry['file']} ---")
    print(f"Artist: {entry['artist']}")
    print(f"Original text length: {len(entry['text'])} chars")
    print(f"Number of tokens: {len(entry['tokens'])}")
    
    # Check special tokens
    tokens = entry['tokens']
    print(f"First token (should be [CLS]=101): {tokens[0]}")
    print(f"Last token (should be [SEP]=102): {tokens[-1]}")
    
    # Verify we can decode back
    decoded = tokenizer.decode(tokens, skip_special_tokens=False)
    original = entry['text']
    
    # Re-tokenize the original text to compare
    re_tokens = tokenizer.encode(original, add_special_tokens=True)
    
    print(f"Tokens match re-tokenization: {tokens == re_tokens}")
    
    # Check if decoded text matches (accounting for special tokens)
    decoded_no_special = tokenizer.decode(tokens, skip_special_tokens=True)
    
    # Normalize whitespace for comparison
    original_normalized = ' '.join(original.split())
    decoded_normalized = ' '.join(decoded_no_special.split())
    
    # Show first 100 chars of each
    print(f"\nOriginal text (first 100 chars): {original[:100]}")
    print(f"Decoded text (first 100 chars): {decoded_no_special[:100]}")
    
    if original_normalized.lower() == decoded_normalized.lower():
        print("✓ Tokenization is REVERSIBLE (text matches after decode)")
    else:
        print("⚠ Tokenization may have issues - text doesn't match exactly")
        print(f"  Original length: {len(original_normalized)}")
        print(f"  Decoded length: {len(decoded_normalized)}")
    
    # Check for empty text case
    if len(original.strip()) == 0:
        print("⚠ WARNING: Empty text file!")
        if tokens == [101, 102]:
            print("  ✓ Correctly tokenized as just [CLS] and [SEP]")
        else:
            print(f"  ✗ Unexpected tokens for empty text: {tokens}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total entries checked: {len(entries)}")
print(f"BERT special tokens:")
print(f"  [CLS] = 101")
print(f"  [SEP] = 102")
print(f"  [PAD] = 0")
print(f"  [UNK] = 100")

