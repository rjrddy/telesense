"""Shared Whisper decoding with model timestamps and overlapping 30s chunks."""
import librosa
from transformers import pipeline


def transcribe_audio(model, processor, device, audio_path, language, word_timestamps):
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    recognizer = pipeline(
        'automatic-speech-recognition', model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, device=device,
    )
    # Chunking keeps recordings longer than Whisper's 30s feature window intact.
    # Timestamp tokens align the review UI to the recording rather than estimating
    # word positions from text length. Predictions still require human review.
    kwargs = {'task': 'transcribe', 'condition_on_prev_tokens': False}
    if language:
        kwargs['language'] = language
    result = recognizer(
        {'raw': audio, 'sampling_rate': sr}, chunk_length_s=30, stride_length_s=5,
        return_timestamps='word' if word_timestamps else True, generate_kwargs=kwargs,
    )
    words = []
    segments = []
    pending = []
    for chunk in result.get('chunks', []):
        text = chunk['text'].strip()
        if not text:
            continue
        start, end = chunk.get('timestamp', (None, None))
        start = max(0.0, min(duration, start if start is not None else (words[-1]['end_time'] if words else 0)))
        end = max(start, min(duration, end if end is not None else duration))
        if not word_timestamps:
            segments.append({'start_time': start, 'end_time': end, 'text': text})
            continue
        word = {'word': text, 'start_time': start, 'end_time': end, 'confidence': None}
        words.append(word)
        pending.append(word)
        if text.endswith(('.', '?', '!')) or end - pending[0]['start_time'] >= 8:
            segments.append({'start_time': pending[0]['start_time'], 'end_time': end, 'text': ' '.join(w['word'] for w in pending)})
            pending = []
    if pending:
        segments.append({'start_time': pending[0]['start_time'], 'end_time': pending[-1]['end_time'], 'text': ' '.join(w['word'] for w in pending)})
    return {'text': result['text'].strip(), 'segments': segments, 'words': words, 'language': language or 'unknown'}
