"""
Data preprocessing for rap lyric classification.
Handles token sequence splitting/truncation to fit BERT's 512 token limit.
"""
import json

# BERT token limits
CLS_TOKEN = 101  # [CLS] token ID
SEP_TOKEN = 102  # [SEP] token ID
MAX_TOKENS = 512  # BERT's maximum sequence length

# Processing strategy: 'truncate' or 'split'
# - 'truncate': Cut sequences to 512 tokens (loses information)
# - 'split': Split long sequences into multiple chunks (preserves all information)
PROCESSING_STRATEGY = 'split'  # Change to 'truncate' if you prefer truncation

# Truncation strategy (only used if PROCESSING_STRATEGY = 'truncate')
# 'start': Keep first 512 tokens (lose end)
#  'end': Keep last 512 tokens (lose beginning)
# 'middle': Keep [CLS], middle tokens, [SEP] (lose both ends)
TRUNCATION_STRATEGY = 'start'


def load_tokenized_data(jsonl_path="tokenized_dataset.jsonl"):
    """
    Load tokenized dataset from JSONL file.
    
    Args:
        jsonl_path: Path to the tokenized dataset JSONL file
        
    Returns:
        List of dictionaries containing tokenized data
    """
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def preprocess_sequences(data, strategy=None, truncation_strategy=None):
    """
    Preprocess token sequences to handle 512 token limit.
    
    Args:
        data: List of dictionaries with 'tokens' and 'artist' keys
        strategy: 'split' or 'truncate' (defaults to PROCESSING_STRATEGY)
        truncation_strategy: 'start', 'end', or 'middle' (defaults to TRUNCATION_STRATEGY)
        
    Returns:
        tuple: (X_texts, y_labels, stats_dict)
        - X_texts: List of token strings (space-separated token IDs)
        - y_labels: List of artist labels
        - stats_dict: Dictionary with preprocessing statistics
    """
    if strategy is None:
        strategy = PROCESSING_STRATEGY
    if truncation_strategy is None:
        truncation_strategy = TRUNCATION_STRATEGY
    
    X_texts = []
    y_labels = []
    truncated_count = 0
    split_count = 0
    total_chunks = 0
    original_lengths = []
    
    for entry in data:
        tokens = entry["tokens"]
        artist = entry["artist"]
        original_lengths.append(len(tokens))
        
        # Handle 512 token limit
        if len(tokens) > MAX_TOKENS:
            if strategy == 'split':
                # Split into multiple chunks
                split_count += 1
                
                # Remove [CLS] and [SEP] if present to split content tokens
                if tokens[0] == CLS_TOKEN and tokens[-1] == SEP_TOKEN:
                    content_tokens = tokens[1:-1]  # Remove [CLS] and [SEP]
                else:
                    content_tokens = tokens
                
                # Split content into chunks of MAX_TOKENS - 2 (leave room for [CLS] and [SEP])
                chunk_size = MAX_TOKENS - 2
                num_chunks = (len(content_tokens) + chunk_size - 1) // chunk_size  # Ceiling division
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min(start_idx + chunk_size, len(content_tokens))
                    chunk_content = content_tokens[start_idx:end_idx]
                    
                    # Add [CLS] and [SEP] to each chunk
                    chunk_tokens = [CLS_TOKEN] + chunk_content + [SEP_TOKEN]
                    
                    # Convert token IDs -> "108 909 1334 ..."
                    token_string = " ".join(str(t) for t in chunk_tokens)
                    
                    X_texts.append(token_string)
                    y_labels.append(artist)  # Same label for all chunks from same song
                    total_chunks += 1
            else:
                # Truncate strategy
                truncated_count += 1
                
                if truncation_strategy == 'start':
                    # Keep first 512 tokens (preserves beginning, loses end)
                    truncated_tokens = tokens[:MAX_TOKENS]
                    
                elif truncation_strategy == 'end':
                    # Keep last 512 tokens (preserves end, loses beginning)
                    truncated_tokens = tokens[-MAX_TOKENS:]
                    
                elif truncation_strategy == 'middle':
                    # Keep [CLS], middle tokens, [SEP] (preserves special tokens, loses both ends)
                    if tokens[0] == CLS_TOKEN and tokens[-1] == SEP_TOKEN:
                        # Keep [CLS], take middle tokens, keep [SEP]
                        truncated_tokens = [tokens[0]] + tokens[1:MAX_TOKENS-1] + [tokens[-1]]
                    else:
                        # If format is unexpected, take middle portion
                        start_idx = (len(tokens) - MAX_TOKENS) // 2
                        truncated_tokens = tokens[start_idx:start_idx + MAX_TOKENS]
                else:
                    # Default to start if strategy is invalid
                    truncated_tokens = tokens[:MAX_TOKENS]
                
                # Convert token IDs -> "108 909 1334 ..."
                token_string = " ".join(str(t) for t in truncated_tokens)
                
                X_texts.append(token_string)
                y_labels.append(artist)
        else:
            # Sequence is already <= 512 tokens, use as-is
            token_string = " ".join(str(t) for t in tokens)
            X_texts.append(token_string)
            y_labels.append(artist)
    
    # Compile statistics
    stats = {
        'original_entries': len(data),
        'split_count': split_count,
        'truncated_count': truncated_count,
        'total_chunks': total_chunks,
        'final_samples': len(X_texts),
        'avg_original_length': sum(original_lengths) / len(original_lengths) if original_lengths else 0,
        'max_original_length': max(original_lengths) if original_lengths else 0,
        'strategy': strategy,
        'truncation_strategy': truncation_strategy if strategy == 'truncate' else None
    }
    
    return X_texts, y_labels, stats


def print_preprocessing_stats(stats):
    """Print preprocessing statistics in a readable format."""
    print(f"Original entries: {stats['original_entries']}")
    if stats['strategy'] == 'split':
        print(f"Entries split (>512 tokens): {stats['split_count']} ({stats['split_count']/stats['original_entries']*100:.1f}%)")
        print(f"Total chunks created: {stats['total_chunks']}")
        print(f"Final training samples: {stats['final_samples']}")
        if stats['split_count'] > 0:
            print(f"Average chunks per split entry: {stats['total_chunks']/stats['split_count']:.1f}")
    else:
        print(f"Entries truncated (>512 tokens): {stats['truncated_count']} ({stats['truncated_count']/stats['original_entries']*100:.1f}%)")
        print(f"Truncation strategy: {stats['truncation_strategy']}")
    print(f"Average original length: {stats['avg_original_length']:.1f} tokens")
    print(f"Max original length: {stats['max_original_length']} tokens")
    print(f"All sequences now <= {MAX_TOKENS} tokens")


if __name__ == "__main__":
    # Example usage
    print("Loading tokenized data...")
    data = load_tokenized_data()
    
    print("Preprocessing sequences...")
    X_texts, y_labels, stats = preprocess_sequences(data)
    
    print("\n" + "="*80)
    print("PREPROCESSING STATISTICS")
    print("="*80)
    print_preprocessing_stats(stats)
    
    print(f"\nSample output:")
    print(f"  Number of samples: {len(X_texts)}")
    print(f"  Number of labels: {len(y_labels)}")
    print(f"  First sample length: {len(X_texts[0].split())} tokens")
    print(f"  First label: {y_labels[0]}")

