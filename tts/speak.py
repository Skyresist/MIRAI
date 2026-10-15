#!/usr/bin/env -S sh -c 'exec "$(dirname "$0")/rvc-venv/bin/python" "$0" "$@"'
"""Text -> Kokoro TTS -> RVC voice conversion -> wav.

Usage:
    ./speak.py -m KosekiBijou.pth "hello chat"
    ./speak.py -m KosekiBijou.pth -f script.txt -o out.wav --pitch 2
"""
import argparse
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
APPLIO = os.path.join(HERE, "Applio")


def parse_args():
    p = argparse.ArgumentParser(description="Kokoro TTS + RVC")
    p.add_argument("text", nargs="?", help="text to speak (or use -f)")
    p.add_argument("-f", "--file", help="read text from a file")
    p.add_argument("-m", "--model", required=True, help="RVC .pth model")
    p.add_argument("-i", "--index", help=".index file (default: same name as model)")
    p.add_argument("-o", "--output", default="speak_output.wav")
    p.add_argument("-v", "--voice", default="af_heart",
                   help="Kokoro voice (af_heart, af_bella, af_nicole, am_adam, ...)")
    p.add_argument("--speed", type=float, default=1.0, help="Kokoro speaking speed")
    p.add_argument("--pitch", type=int, default=6,
                   help="semitone shift before RVC (+12 male->female, -12 female->male)")
    p.add_argument("--index-rate", type=float, default=0.75,
                   help="0-1, how strongly to pull toward the model's timbre")
    p.add_argument("--protect", type=float, default=0.5,
                   help="0-0.5, protects consonants/breaths from artifacts")
    p.add_argument("--f0", default="rmvpe", choices=["rmvpe", "fcpe", "crepe"])
    p.add_argument("--keep-tts", action="store_true", help="also save the raw Kokoro wav")
    a = p.parse_args()
    if a.file:
        with open(a.file, encoding="utf-8") as fh:
            a.text = fh.read()
    if not a.text:
        p.error("give text or -f FILE")
    return a


def tts(text, voice, speed, out_path):
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipe = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M")
    chunks = [audio for _, _, audio in pipe(text, voice=voice, speed=speed)]
    sf.write(out_path, np.concatenate(chunks), 24000)


def main():
    a = parse_args()
    model = os.path.abspath(a.model)
    index = os.path.abspath(a.index) if a.index else os.path.splitext(model)[0] + ".index"
    if not os.path.isfile(index):
        print(f"no index found at {index}, continuing without one")
        index = ""
    output = os.path.abspath(a.output)

    fd, tts_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    print(f"[1/2] Kokoro ({a.voice})...")
    tts(a.text, a.voice, a.speed, tts_wav)
    if a.keep_tts:
        import shutil
        shutil.copy(tts_wav, os.path.splitext(output)[0] + "_tts.wav")

    # Applio resolves its models relative to its own folder
    os.chdir(APPLIO)
    sys.path.insert(0, APPLIO)
    from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline
    if not os.path.isfile("rvc/models/predictors/rmvpe.pt") or \
       not os.path.isfile("rvc/models/embedders/contentvec/pytorch_model.bin"):
        prequisites_download_pipeline(False, True, False)
    from rvc.infer.infer import VoiceConverter

    print(f"[2/2] RVC ({os.path.basename(model)})...")
    VoiceConverter().convert_audio(
        audio_input_path=tts_wav,
        audio_output_path=output,
        model_path=model,
        index_path=index,
        pitch=a.pitch,
        f0_method=a.f0,
        index_rate=a.index_rate,
        protect=a.protect,
        volume_envelope=1.0,
        split_audio=len(a.text) > 400,
    )
    os.remove(tts_wav)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
