import asyncio
import pathlib
import sys

import edge_tts


async def main():
    # Parse command line arguments
    tts_file = str(sys.argv[1])
    text = str(sys.argv[2])
    voice = str(sys.argv[3])
    rate = int(sys.argv[4])
    output_file = str(sys.argv[5])

    rates = f"+{rate}%" if rate >= 0 else f"{rate}%"
    if tts_file and pathlib.Path(tts_file).exists():
        text = ""
        try:
            with pathlib.Path(tts_file).open(encoding="utf-8") as file:
                text = file.read()
        except UnicodeDecodeError:
            with pathlib.Path(tts_file).open() as file:
                text = file.read()
    await edge_tts.Communicate(text, voice, rate=rates).save(output_file)
    # print(f"TTS with {voice} completed. Output TTS file: '{output_file}'")


if __name__ == "__main__":
    asyncio.run(main())
