"""
Inference script for temporal localization model.
"""

import argparse
import json
from telesense_ml.supervised.inference.temporal_predictor import TemporalPredictor


def main():
    """
    Main.
    Input:
        None

    Output:
        None
    """
    parser = argparse.ArgumentParser(description='Run temporal localization inference')
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
        '--segments-only',
        action='store_true',
        help='Output only segments (simplified)'
    )
    
    args = parser.parse_args()
    
    # Check that at least one input is provided
    if not any([args.audio, args.video, args.text]):
        print("Error: At least one input (--audio, --video, or --text) must be provided")
        return
    
    # Initialize predictor
    print(f"Loading model from {args.checkpoint}...")
    predictor = TemporalPredictor(args.checkpoint)
    
    # Make prediction
    print("Running temporal localization...")
    
    if args.segments_only:
        segments = predictor.predict_segments_only(
            audio_path=args.audio,
            video_path=args.video,
            text=args.text
        )
        
        print("\n" + "="*50)
        print("Detected Temporal Segments")
        print("="*50)
        for start, end, concern_type, confidence in segments:
            concern_name = predictor.class_names[concern_type] if predictor.class_names else f"Concern_{concern_type}"
            print(f"Frame {start}-{end}: {concern_name} (confidence: {confidence:.4f})")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(segments, f, indent=2)
            print(f"\nSegments saved to {args.output}")
    else:
        result = predictor.predict(
            audio_path=args.audio,
            video_path=args.video,
            text=args.text
        )
        
        # Print summary
        print("\n" + "="*50)
        print("Temporal Localization Results")
        print("="*50)
        print(f"Total frames analyzed: {result['num_frames']}")
        print(f"Detected segments: {len(result['segments'])}")
        
        if result['segments']:
            print("\nConcern Segments:")
            for start, end, concern_type, confidence in result['segments']:
                concern_name = predictor.class_names[concern_type] if predictor.class_names else f"Concern_{concern_type}"
                start_time = start / 30.0  # Assuming 30 fps
                end_time = end / 30.0
                print(f"  [{start_time:.2f}s - {end_time:.2f}s] {concern_name} (confidence: {confidence:.4f})")
        
        # Save output if requested
        if args.output:
            # Convert numpy arrays to lists for JSON
            output_dict = {
                'num_frames': result['num_frames'],
                'segments': result['segments'],
                'frame_predictions': result['frame_predictions']
            }
            with open(args.output, 'w') as f:
                json.dump(output_dict, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

