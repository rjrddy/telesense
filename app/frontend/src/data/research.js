export const models = [
  {
    id: "speech",
    name: "Whisper base",
    role: "Speech → transcript",
    icon: "file",
    tone: "blue",
    checkpoint: "openai/whisper-base",
    facts: [
      "74M parameters",
      "Encoder–decoder Transformer",
      "30-second overlapping chunks",
    ],
    description:
      "Converts the recording into text with model-derived word timestamps. Those timestamps connect transcript passages to playback.",
    training:
      "Pretrained on a large, multilingual collection of labeled speech. This demo uses the base checkpoint without telehealth fine-tuning.",
    limitation:
      "Noise, accents, overlapping speakers, and silence can cause omissions or invented words. Errors propagate into the language analysis. Timestamps are estimates from the model, not manually verified alignments.",
    source: "https://huggingface.co/openai/whisper-base",
  },
  {
    id: "voice",
    name: "Wav2Vec2 · SUPERB",
    role: "Audio → vocal emotion",
    icon: "wave",
    tone: "teal",
    checkpoint: "superb/wav2vec2-base-superb-er",
    facts: [
      "16 kHz mono input",
      "3-second windows · 0.5-second hop",
      "4 emotion classes",
    ],
    description:
      "Scores short, overlapping windows for neutral, happy, angry, and sad vocal expression. It analyzes the audio directly rather than the transcript.",
    training:
      "A Wav2Vec2 speech representation with an emotion-recognition head evaluated on the four-class IEMOCAP task.",
    limitation:
      "Acted speech and brief utterances differ from clinical conversation. Emotion is not a mental-health condition, and vocal style, language, recording quality, and speaker differences affect predictions.",
    source: "https://huggingface.co/superb/wav2vec2-base-superb-er",
  },
  {
    id: "language",
    name: "RoBERTa · GoEmotions",
    role: "Transcript → language emotion",
    icon: "text",
    tone: "amber",
    checkpoint: "SamLowe/roberta-base-go_emotions",
    facts: [
      "28 output labels",
      "Multi-label text classification",
      "English Reddit training data",
    ],
    description:
      "Assigns emotion scores to transcript passages. Several labels can apply to one passage, including sadness, fear, curiosity, joy, and neutral.",
    training:
      "RoBERTa base fine-tuned on GoEmotions, a corpus of annotated Reddit comments. Conversation transcripts are a different input domain.",
    limitation:
      "Sarcasm, quoted speech, speaker identity, and transcription errors can change the result. The model has no access to a person’s history or clinical assessment.",
    source: "https://huggingface.co/SamLowe/roberta-base-go_emotions",
  },
];

export const datasets = [
  {
    name: "LibriSpeech",
    type: "Read speech · audio + transcripts",
    recommendation: "Best next transcription sample",
    status: "Attribution required",
    license: "CC BY 4.0",
    description:
      "Public-domain audiobook readings packaged as an ASR corpus. Useful for comparing recognized words with reference transcripts and testing clean versus challenging audio.",
    caveat:
      "Not an emotion or clinical dataset. A reading cannot serve as evidence of a diagnosis. Include corpus attribution and identify the original clip.",
    url: "https://www.openslr.org/12",
    tone: "blue",
  },
  {
    name: "CREMA-D",
    type: "Acted emotion · audio + video",
    recommendation: "Best next emotion comparison",
    status: "License obligations to implement",
    license: "ODbL 1.0 + DbCL 1.0 for contents",
    description:
      "7,442 clips from 91 actors using six emotional expressions. Matched sentences would let visitors compare vocal delivery while keeping the words similar.",
    caveat:
      "Preserve source attribution and license notices, and review database/share-alike obligations for the selected release. The repository also requests a user registration form. Acted labels are not diagnoses.",
    url: "https://github.com/CheyneyComputerScience/CREMA-D",
    tone: "teal",
  },
  {
    name: "RAVDESS",
    type: "Acted speech/song · audio + video",
    recommendation: "Useful multimodal alternative",
    status: "Noncommercial / share-alike",
    license: "CC BY-NC-SA 4.0",
    description:
      "Recordings of 24 actors, with matched statements and different emotions and intensities. Useful for demonstrating agreement and disagreement across modalities.",
    caveat:
      "The publisher offers separate commercial licenses. Resolve intended portfolio use and redistribution terms before hosting recordings; do not assume every public website qualifies as noncommercial.",
    url: "https://zenodo.org/records/1188976",
    tone: "amber",
  },
  {
    name: "DAIC-WOZ",
    type: "Interview research · audio + transcripts + facial features",
    recommendation: "Private research candidate",
    status: "Restricted access",
    license: "Application and signed agreement",
    description:
      "189 interviews collected for research into psychological distress, with transcripts and behavioral features. More relevant to clinical research than acted emotion clips.",
    caveat:
      "The provider limits distribution to academics and nonprofit researchers because of consent constraints. Do not publish participant recordings in the portfolio demo on the basis of research access.",
    url: "https://dcapswoz.ict.usc.edu/",
    tone: "rose",
  },
];
