# Sample recordings and candidate sources

Reviewed October 25, 2026. The public library contains eight recordings: three approved local excerpts, three CREMA-D videos, and two LibriSpeech audio clips. RAVDESS and DAIC-WOZ remain candidates, not hosted recordings.

| Source | Useful addition | Published terms | Next step |
| --- | --- | --- | --- |
| [LibriSpeech / OpenSLR 12](https://www.openslr.org/12) | Read speech with reference transcripts; compare ASR output against known text | CC BY 4.0 | Select short clips, preserve identifiers and corpus attribution, document any transcoding. A transcript reference is not a clinical label. |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | The same sentence expressed with different acted emotions; audio/video comparison | ODbL 1.0 for the database; DbCL 1.0 for individual contents | Implement attribution, license notices, and applicable database/share-alike obligations before redistribution. The repository requests a user-registration form. Do not describe acted emotion labels as clinical ground truth. |
| [RAVDESS](https://zenodo.org/records/1188976) | Matched speech statements with different emotions/intensities, with audio and video | CC BY-NC-SA 4.0; separate commercial licenses offered | Resolve whether the intended use fits the terms before hosting media. Preserve attribution and share-alike requirements where applicable. |
| [DAIC-WOZ](https://dcapswoz.ict.usc.edu/) | Distress-research interviews with transcripts, audio, and facial features | Application/signature; distribution limited to academic and nonprofit researchers because of consent constraints | Consider for approved private research only. Research access does not establish permission to publish participant recordings in a portfolio. |

## Suggested first set

1. Two or three short LibriSpeech utterances, including a challenging recording and its reference transcript.
2. A small CREMA-D set with matched text and contrasting acted expressions, after implementing source/license obligations.
3. Keep RAVDESS as an alternative pending the appropriate reuse terms. Keep DAIC-WOZ out of public demo assets unless separately authorized for that use.

For each addition record: author/corpus, source URL, original clip ID, license URL/version, reference label or transcript, and modifications (cuts, resampling, transcoding). Store corpus labels separately from model outputs. Run the actual pretrained pipeline and label its output as saved analysis. Do not infer or attach a diagnosis to a speaker from these samples.

The in-app **Models & context → Sample data sources** view exposes this shortlist and links to the primary publishers.

## Published external set

- CREMA-D: `1001_ITH_NEU_XX`, `1001_ITH_HAP_XX`, `1001_ITH_SAD_XX` — the same sentence, performed with neutral, happy, and sad expressions. Full clips are converted to browser-compatible MP4.
- LibriSpeech: `1272-128104-0000` and `1272-135031-0000` — full dev-clean utterances from the Hugging Face internal-testing mirror, with corpus reference text retained.

The UI includes attribution, source-label explanations, reference transcripts, modification details, and license downloads. The CREMA-D subset/catalog is separately distributed under ODbL 1.0; individual contents retain DbCL 1.0. LibriSpeech retains CC BY 4.0. See `app/frontend/public/demo/ATTRIBUTION.md` and the `licenses/` directory. This does not relicense unrelated recordings or the application source.

Run `python scripts/fetch_demo_sources.py` to fetch and checksum-verify source files, and `python scripts/build_demo_samples.py` in the configured CPU environment to generate results. Do not substitute corpus labels or reference transcripts for actual model predictions.
