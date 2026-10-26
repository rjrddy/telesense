# Candidate sample recordings

Reviewed October 25, 2026. These sources are researched candidates, not media already added to the public library. Existing published recordings remain the approved local test_data excerpts.

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
