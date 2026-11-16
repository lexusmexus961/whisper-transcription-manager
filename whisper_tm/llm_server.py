"""
Start the local llama.cpp server with values from config.

``./start-llm.sh`` only calls this module — so the startup command
is at exactly one place (core/llama_server.start_command) and tuning happens
via the ``llama`` config section without regenerating the script.

Call:
    python -m whisper_tm.llm_server           # start server
    python -m whisper_tm.llm_server --show    # output command only
    python -m whisper_tm.llm_server --check # situation report (like health)

Additional arguments are passed through to llama-server.
"""

from __future__ import annotations

import json
import os
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from whisper_tm.config import config

    config.load(None)
    from whisper_tm.core import llama_server

    if "--check" in argv:
        print(json.dumps(llama_server.check(), indent=2, ensure_ascii=False))
        return 0

    command = llama_server.start_command()
    if not command:
        if not llama_server.binary():
            print("No llama-server found. Install with:\n  ./setup.sh --llama", file=sys.stderr)
        else:
            print(
                "No GGUF model found. Place a .gguf file at\n"
                f"  {llama_server.base_directory() / 'models' / 'gguf'}\n"
                "or set llama.model in config.",
                file=sys.stderr,
            )
        return 1

    passed_through = [a for a in argv if a != "--show"]
    command += passed_through

    if "--show" in argv:
        print(" \\\n  ".join(command))
        return 0

    model = llama_server.model_file()
    threads, reason = llama_server.recommended_threads()
    print(f"Model:    {model.name if model else '?'}")
    print(f"Endpoint: {llama_server.endpoint()}")
    print(f"Threads:  {threads} ({reason})")
    print(
        f"Context:  {config.get('llama', 'ctx_size', default=32768)}"
        f" · Ubatch {config.get('llama', 'ubatch_size', default=512)}"
    )
    print(
        "Server uses VRAM while running — exit with Ctrl+C before\na transcription needs the GPU.\n"
    )
    sys.stdout.flush()
    if os.name == "nt":
        # Windows has no exec: os.execv joins the arguments unquoted (a
        # path with a space falls apart) and the prompt comes back at
        # once. Ctrl+C reaches both through the console.
        import subprocess

        proc = subprocess.Popen(command)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            return proc.wait()
    # exec: llama-server inherits the terminal, Ctrl+C acts directly on it
    os.execv(command[0], command)
    return 0  # never reached


if __name__ == "__main__":
    raise SystemExit(main())
