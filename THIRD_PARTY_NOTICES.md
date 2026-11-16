# Third-party notices

Whisper Transcription Manager is released under the MIT License ([LICENSE](LICENSE)). This
repository also contains the following third-party material, each under its
own license.

## Plyr 3.7.8

Files: `whisper_tm/static/vendor/plyr/` (`plyr.min.js`, `plyr.css`,
`plyr.svg`; the license is next to them as `LICENSE.md`)
Source: https://github.com/sampotts/plyr

```
The MIT License (MIT)

Copyright (c) 2017 Sam Potts

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Feather Icons

Files: the icon sprite in `whisper_tm/static/index.html` (`<symbol id="i-…">`),
drawn after Feather's stroke geometry
Source: https://github.com/feathericons/feather

```
The MIT License (MIT)

Copyright (c) 2013-2023 Cole Bemis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ReDimNet

Files: `whisper_tm/vendor/redimnet/` — the model code of
[IDRnD/ReDimNet](https://github.com/IDRnD/ReDimNet) at commit `dd9f519`, its
imports made relative (the license is next to it as `LICENSE`). Its weights are
not part of this repository: they are downloaded once from the project's
release.

Two of its files hold code from other projects under the Apache License 2.0,
whose text is next to them as `LICENSE-APACHE-2.0`; both are unchanged:

- `layers/poolings.py` — from [WeSpeaker](https://github.com/wenet-e2e/wespeaker),
  Copyright (c) 2021 Shuai Wang (the file keeps its header).
- `layers/attention.py` — the attention module and the activation classes are
  copied from [Hugging Face transformers](https://github.com/huggingface/transformers)
  v4.26.1 (`models/wav2vec2/modeling_wav2vec2.py`, Copyright 2021 The Fairseq
  Authors and the HuggingFace Inc. team; `activations.py`, Copyright 2020 The
  HuggingFace Team). All rights reserved by them; the notice at the top of
  the file names both.

Neither project ships a NOTICE file.

The rest of the folder:

```
MIT License

Copyright (c) 2024 ID R&D, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Not part of this repository

The Python packages, the models and the programs the setup installs or
downloads (PyTorch, WhisperX, pyannote.audio, audio-separator, FFmpeg,
llama.cpp, …) come with their own licenses and terms — see docs/GUIDE.md:
Models and their licenses. One of them is non-commercial: `diffq`
(CC BY-NC 4.0), which audio-separator installs on every path although this
app doesn't use it; and the voice model's weights are trained on VoxBlink2
(CC BY-NC-SA 4.0, no commercial use).
