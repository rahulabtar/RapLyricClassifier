import os
import re
from pathlib import Path

# List of allowed artists
artist_list = ["Kanye West", "Drake", "Jayz", "Kendrick Lamar", "Ice Cube",
               "Snoop Dogg", "Eminem", "Travis Scott", "Lil Wayne", "Dr. Dre"]

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


# Example usage
clean_lyrics(artist_list)
