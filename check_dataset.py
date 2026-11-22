
from pathlib import Path

# Locate the extracted GTZAN folder
project_path = Path(__file__).resolve().parent
gtzan_path = project_path.parent / "GTZAN"

print("Checking dataset at:", gtzan_path)

if not gtzan_path.exists():
    print("GTZAN folder not found!")
    print("Please check the folder name and location.")

else:
    # Find the folder containing genre audio files
    genre_path = gtzan_path / "genres_original"

    if not genre_path.exists():
        matches = list(gtzan_path.rglob("genres_original"))

        if matches:
            genre_path = matches[0]
        else:
            print("genres_original folder not found!")
            print("Inspect the extracted folder structure.")
            raise SystemExit

    genres = sorted(
        folder for folder in genre_path.iterdir()
        if folder.is_dir()
    )

    total_tracks = 0

    print("\nGenres found:")

    for genre in genres:
        tracks = list(genre.glob("*.wav"))
        total_tracks += len(tracks)
        print(f"{genre.name}: {len(tracks)} audio files")

    print("\nTotal genres:", len(genres))
    print("Total WAV files:", total_tracks)
    print("Dataset verification completed!")
