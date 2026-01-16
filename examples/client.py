import requests
import json

API_URL = "http://localhost:8899"

LYRICS = """[Intro]

[Verse]
The sun creeps in across the floor
I hear the traffic outside the door
The coffee pot begins to hiss
It is another morning just like this

[Prechorus]
The world keeps spinning round and round
Feet are planted on the ground
I find my rhythm in the sound

[Chorus]
Every day the light returns
Every day the fire burns
We keep on walking down this street
Moving to the same steady beat
It is the ordinary magic that we meet

[Verse]
The hours tick deeply into noon
Chasing shadows,chasing the moon
Work is done and the lights go low
Watching the city start to glow

[Bridge]
It is not always easy,not always bright
Sometimes we wrestle with the night
But we make it to the morning light

[Chorus]
Every day the light returns
Every day the fire burns
We keep on walking down this street
Moving to the same steady beat

[Outro]
Just another day
Every single day"""

TAGS = "instrumental,guitar"


def generate_with_progress(run_id: int, max_audio_length_ms: int = 240_000):
    """Generate music with streaming progress updates."""
    save_path = f"./stream_{run_id}.mp3"

    print(f"\n{'='*60}")
    print(f"Starting generation {run_id} -> {save_path}")
    print(f"{'='*60}")

    payload = {
        "lyrics": LYRICS,
        "tags": TAGS,
        "max_audio_length_ms": max_audio_length_ms,
        "save_path": save_path,
    }

    response = requests.post(
        f"{API_URL}/generate/stream",
        json=payload,
        stream=True,
        headers={"Accept": "text/event-stream"},
    )

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = json.loads(line[6:])

                if data["type"] == "progress":
                    pct = data["progress_pct"]
                    frames = data["generated_frames"]
                    elapsed_ms = data["elapsed_ms"]
                    # Print progress on same line
                    print(f"\r[{run_id}] Progress: {pct:5.1f}% | Frames: {frames:4d} | Audio: {elapsed_ms/1000:.1f}s", end="", flush=True)

                elif data["type"] == "complete":
                    print(f"\n[{run_id}] Complete! Saved to: {data['save_path']}")

                elif data["type"] == "error":
                    print(f"\n[{run_id}] Error: {data['error']}")
                    return False

    return True


def main():
    num_generations = 10
    max_audio_length_ms = 240_000  # 4 minutes

    print(f"Generating {num_generations} tracks back-to-back")
    print(f"Max audio length: {max_audio_length_ms/1000:.0f}s per track")

    successful = 0
    for i in range(1, num_generations + 1):
        if generate_with_progress(i, max_audio_length_ms):
            successful += 1

    print(f"\n{'='*60}")
    print(f"Done! {successful}/{num_generations} generations completed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
