# Public demo recording credits

The three original samples (conversation, listening, reflection) are excerpts from the owner's approved local test_data collection. Their reuse permission is separate from the corpus licenses below.

## CREMA-D three-clip subset

Source: Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D), Cao, Cooper, Keutmann, Gur, Nenkova & Verma (2014).
Repository: https://github.com/CheyneyComputerScience/CREMA-D
Paper: https://doi.org/10.1109/TAFFC.2014.2336244

Contains information from CREMA-D, made available under the Open Database License (ODbL): https://opendatacommons.org/licenses/odbl/1-0/
Individual contents are licensed under the Database Contents License (DbCL): https://opendatacommons.org/licenses/dbcl/1-0/

This selected subset and its source records are available under ODbL 1.0 in `crema-d-subset.json`. Copies of the ODbL and DbCL are in `licenses/`. These licenses apply to the CREMA-D subset, not to the unrelated original recordings or application source code.

Clips: 1001_ITH_NEU_XX, 1001_ITH_HAP_XX, 1001_ITH_SAD_XX. Actor 1001; sentence ITH; neutral, happy, and sad performances; unspecified intensity. These are acted expressions, not clinical labels.

Changes: the full FLV videos were transcoded to H.264/AAC MP4, a thumbnail was extracted, and audio was downmixed/resampled to mono 16 kHz for analysis. No diagnostic label was assigned to the actor. Actual predicted model outputs are saved separately from source labels.

## LibriSpeech

Corpus authors: Vassil Panayotov, Guoguo Chen, Daniel Povey & Sanjeev Khudanpur (2015). Recordings originated with LibriVox readers.
Original source: https://www.openslr.org/12
Mirror used: https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy
License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
A copy of the legal text is in `licenses/CC-BY-4.0.txt`.

Clips: 1272-128104-0000 and 1272-135031-0000 from dev-clean. Source reference transcripts are retained in each sample's attribution metadata. These are audiobook readings, not patient recordings or clinical ground truth.

Changes: full utterances were extracted from the mirrored Parquet corpus and transcoded to MP3 for playback and mono 16 kHz WAV for analysis. Model-produced transcripts and concern flags are derived outputs and may differ from the reference text.

No corpus author, reader, or actor is presented as endorsing Telesense. Per-file source URLs, checksums, reference text, and modifications are retained in the manifest and exported results.
