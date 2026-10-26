"""
Inference script for unsupervised temporal localization model.
"""

import argparse
import json
import os

from telesense_ml.unsupervised.inference import UnsupervisedPredictor


def main():
    """
    Main.
    Input:
        None

    Output:
        None
    """
    parser = argparse.ArgumentParser(description='Run unsupervised temporal localization inference')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--audio',
        type=str,
        default=None,
        help='Path to audio file'
    )
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Path to video file'
    )
    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help='Text input'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output (JSON format)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Anomaly threshold (if None, use adaptive)'
    )
    
    args = parser.parse_args()
    
    # Check that at least one input is provided
    if not any([args.audio, args.video, args.text]):
        print("Error: At least one input (--audio, --video, or --text) must be provided")
        return
    
    # Initialize predictor
    print(f"Loading model from {args.checkpoint}...")
    predictor = UnsupervisedPredictor(args.checkpoint)
    
    # Make prediction
    print("Running unsupervised anomaly detection...")
    result = predictor.predict(
        audio_path=args.audio,
        video_path=args.video,
        text=args.text,
        threshold=args.threshold
    )
    
    # Print summary
    print("\n" + "="*50)
    print("Unsupervised Anomaly Detection Results")
    print("="*50)
    print(f"Total frames analyzed: {result['num_frames']}")
    print(f"Anomaly threshold: {result['threshold']:.4f}")
    print(f"Flagged frames: {result['flagged_frames'].sum()}")
    print(f"Anomaly segments: {len(result['segments'])}")
    
    if result['segments']:
        print("\nAnomaly Segments:")
        for start, end, confidence in result['segments']:
            start_time = start / 30.0  # Assuming 30 fps
            end_time = end / 30.0
            print(f"  [{start_time:.2f}s - {end_time:.2f}s] (anomaly score: {confidence:.4f})")
    
    # Save output if requested
    if args.output:
        output_dict = {
            'num_frames': result['num_frames'],
            'threshold': result['threshold'],
            'segments': result['segments'],
            'anomaly_scores': result['anomaly_scores'].tolist(),
            'flagged_frames': result['flagged_frames'].tolist()
        }
        with open(args.output, 'w') as f:
            json.dump(output_dict, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

