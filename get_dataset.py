import pandas as pd
import lyricsgenius
import os
from dotenv import load_dotenv
import json
from transformers import AutoTokenizer
from pathlib import Path
import re

exlude_terms = ["(Remix)", "(Live)", "(Demo)", "(Acoustic)", "(Instrumental)", "(Interlude)", 
                             "(Version)", "(Reference)", "interview", "(Explicit)", "(Clean)", "Reference)", 
                             "MTV", "Radio Edit", "Demo)", "Version)", "Reference", "[V2]", "[V3]", "[Remix]", "Remix)",
                             "mix"]

def getArtistSongs(artist, exclude_terms=exlude_terms):
    titles = []
    lyrics = []
    artist_genius = genius.search_artist(artist, max_songs = None, sort='title')
    songs = artist_genius.songs
    genius.excluded_terms = exclude_terms
    print(f"Fetched {len(songs)} songs for artist {artist_genius.name}")
    for song in songs:
        titles.append(song.title)
        lyrics.append(song.lyrics)
        if os.path.exists(f"lyrics/{artist}/{song.title}.txt"):
            print(f"Song {song.title} already exists. Skipping...")
            continue
        elif any(term.lower() in song.title.lower() for term in exclude_terms):
            print(f"Song {song.title} contains excluded terms. Skipping...")
            continue
        elif len(song.lyrics) < 100:
            print(f"Song {song.title} has insufficient lyrics. Skipping...")
            continue
        else:
            try:
                with open(f"lyrics/{artist}/{song.title}.txt", "w", encoding="utf-8") as f:
                    f.write(song.lyrics)
                    print(f"Saved lyrics for song {song.title}")
            except Exception as e:
                print(f"Error saving lyrics for song {song.title}: {e}")

def clean_lyrics(artist_list):
    for artist in artist_list:

        artist_folder = Path("lyrics") / artist
        cleaned_folder = Path("cleaned_lyrics") / artist
        cleaned_folder.mkdir(parents=True, exist_ok=True)

        section_regex = re.compile(r"\[(.*?)\]", re.MULTILINE)

        for filename in os.listdir(artist_folder):
            if not filename.endswith(".txt"):
                continue

            with open(artist_folder / filename, "r", encoding="utf-8") as f:
                lyrics_text = f.read()

            lines = lyrics_text.split("\n")

            # Only track the main artist’s lines
            collected = []

            current_artists = [artist]  # default

            for line in lines:
                stripped = line.strip()

                # Detect section headers
                header_match = section_regex.match(stripped)
                if header_match:
                    header = header_match.group(1)

                    if ":" in header:
                        _, artists_part = header.split(":", 1)
                        artists = re.split(r"&|,| and ", artists_part)
                        artists = [a.strip() for a in artists if a.strip()]
                    else:
                        artists = [artist]

                    # Keep only artists from the whitelist
                    artists = [a for a in artists if a in artist_list]

                    # If main artist is NOT in the list of current artists → they shouldn’t get this verse
                    if artist in artists:
                        current_artists = [artist]
                    else:
                        current_artists = []  # main artist does not sing this verse

                    continue

                # Add lyric lines ONLY if main artist is currently active
                if stripped and artist in current_artists:
                    collected.append(stripped)

            # Save cleaned lyrics into ONLY the main artist's folder
            base_name = Path(filename).stem
            out_path = cleaned_folder / f"{base_name}.txt"

            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(collected))

            print(f"Finished processing {artist}: {filename}")

#Get the key from the .env file and print
load_dotenv() 
key =  str(os.getenv("API_KEY"))
print("Got key from .env:", key)
client_access_token = os.getenv("API_KEY")


#creaete directories to store lyrics
artist_list = ["Kanye West", "Drake", "Jayz", "Kendrick Lamar", "Ice Cube", "Snoop Dogg", "Eminem", "Travis Scott", "Lil Wayne", "Dr. Dre"]

genius = lyricsgenius.Genius(client_access_token, remove_section_headers=False, skip_non_songs=True, timeout=30, retries=3)
genius.skip_non_primary_artists = True
for artist in artist_list:
    os.makedirs(os.path.join('lyrics', artist), exist_ok=True)
    getArtistSongs(artist)

#clean the lyrics and organize 
clean_lyrics(artist_list)

# Tokenize the cleaned lyrics and save to JSONL
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
