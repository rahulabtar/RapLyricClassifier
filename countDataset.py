from pathlib import Path

# Path to the main folder
input_root = Path("cleaned_lyrics")

# Dictionary to hold results
artist_stats = {}

# Loop over each artist folder
for artist_folder in input_root.iterdir():
    if artist_folder.is_dir():
        song_count = 0
        word_count = 0

        # Loop over each text file in the artist folder
        for song_file in artist_folder.glob("*.txt"):
            song_count += 1
            try:
                with song_file.open("r", encoding="utf-8") as f:
                    text = f.read()
                    word_count += len(text.split())
            except Exception as e:
                print(f"Error reading {song_file}: {e}")

        artist_stats[artist_folder.name] = {
            "songs": song_count,
            "words": word_count
        }

# Print results
total_songs = 0
total_words = 0
for artist, stats in artist_stats.items():
    print(f"{artist}: {stats['songs']} songs, {stats['words']} words")
    total_songs += stats['songs']
    total_words += stats['words']

print(f"\nTotal: {total_songs} songs, {total_words} words")
