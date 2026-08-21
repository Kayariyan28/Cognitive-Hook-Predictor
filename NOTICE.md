# Third-party notices

Copyright (c) 2026 Karan Chandra Dey.

The MIT License in [`LICENSE`](LICENSE) applies only to original SignalFrame
source code and documentation owned by the copyright holder. It does not
relicense external software, model code, model weights, fonts, scientific data,
or trademarks. Each third-party component remains subject to its own license
and terms.

## Models and model code

- **TRIBE v2** — Meta Platforms, Inc. The official repository and model weights
  are distributed under **CC BY-NC 4.0**. They are downloaded separately and
  are not granted commercial-use rights by SignalFrame's MIT License. Review
  the [official TRIBE v2 license](https://github.com/facebookresearch/tribev2/blob/main/LICENSE)
  before use.
- **V-JEPA 2 / V-JEPA 2.1** — Meta Platforms, Inc. The worker pins the official
  V-JEPA 2 source and checkpoint identities. Most upstream source is MIT;
  identified files are Apache-2.0, and checkpoint terms must be checked for the
  exact artifact. Review the [official V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2)
  and the model card before use.
- **NanoLLaVA 1.5 8-bit (MLX community conversion)** — downloaded separately
  from its model repository and recorded by the worker as Apache-2.0. The
  original model, conversion, dependencies, and weights may carry distinct
  notices; review the complete model card and included license files.
- **Audio Spectrogram Transformer AudioSet checkpoint** — downloaded separately
  and recorded by the worker as BSD-3-Clause. Review the exact checkpoint model
  card and upstream implementation notices.
- **Whisper (MLX community conversion)** — the optional transcript branch pins
  an `mlx-community` Whisper conversion by immutable revision and downloads it
  separately. OpenAI's original Whisper model and code are **MIT**; the MLX
  conversion, its dependencies, and its weights may carry distinct notices.
  Review the exact model card and included license files before use. No
  transcript model is bundled in this repository.

No model weights, creator uploads, generated cortical tensors, or local model
caches are intended to be committed to this repository.

## Scientific assets

The browser-ready cortical surface and atlas mappings under
`public/assets/brain/` are derived from upstream `fsaverage5`, Nilearn,
FreeSurfer, NITRC, and Destrieux atlas resources. Source URLs and content hashes
are embedded in the adjacent JSON metadata and the build scripts. The bundled
surface is a **modified derivative**, produced by averaging pial and inflated
coordinates, shifting the hemispheres, and repacking the geometry into the
SignalFrame binary layout; it must not be represented as an original
FreeSurfer distribution. The required FreeSurfer Part B terms and redistribution
preface are included in
[`third_party/licenses/FREESURFER-LICENSE.txt`](third_party/licenses/FREESURFER-LICENSE.txt).
The Nilearn BSD 3-Clause terms are included in
[`third_party/licenses/NILEARN-BSD-3-CLAUSE.txt`](third_party/licenses/NILEARN-BSD-3-CLAUSE.txt).
These assets remain subject to their upstream data and software terms; the
SignalFrame MIT License does not replace those terms.

## Fonts and software dependencies

- Inter (`public/assets/inter.woff2`, SHA-256
  `c940764593d0fe5d596be327ca7558855e018039fb78509aa21921fd3644c3e4`) is
  copyright the Inter Project Authors and distributed under SIL OFL 1.1. The
  complete notice and license are in
  [`third_party/licenses/INTER-OFL-1.1.txt`](third_party/licenses/INTER-OFL-1.1.txt).
- Space Grotesk (`public/assets/space-grotesk.woff2`, SHA-256
  `a0d054c4af557de20afd6ca59f47ab353bcaec49c63ff04b6c9d39d0f8910557`) is
  copyright the Space Grotesk Project Authors and distributed under SIL OFL
  1.1. The complete notice and license are in
  [`third_party/licenses/SPACE-GROTESK-OFL-1.1.txt`](third_party/licenses/SPACE-GROTESK-OFL-1.1.txt).
- JavaScript and Python dependencies retain the licenses declared by their
  packages. Exact direct versions are recorded in `package-lock.json` and
  `backend/requirements.txt`.

Meta, TRIBE, V-JEPA, Hugging Face, FreeSurfer, Nilearn, and other names may be
trademarks of their respective owners. SignalFrame is an independent project
and is not endorsed by or affiliated with those organizations.
