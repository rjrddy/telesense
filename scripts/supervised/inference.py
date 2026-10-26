"""
Inference script for multimodal behavioral model.
"""

import argparse
from telesense_ml.supervised.inference import Predictor


def main():
    """
    Main.
    Input:
        None

    Output:
        None
    """
    parser = argparse.ArgumentParser(description='Run inference on multimodal behavioral model')
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
    
    args = parser.parse_args()
    
    # Check that at least one input is provided
    if not any([args.audio, args.video, args.text]):
        print("Error: At least one input (--audio, --video, or --text) must be provided")
        return
    
    # Initialize predictor
    print(f"Loading model from {args.checkpoint}...")
    predictor = Predictor(args.checkpoint)
    
    # Make prediction
    print("Running inference...")
    result = predictor.predict(
        audio_path=args.audio,
        video_path=args.video,
        text=args.text
    )
    
    # Print results
    print("\n" + "="*50)
    print("Prediction Results")
    print("="*50)
    print(f"Predicted Class: {result['predicted_class']}")
    if 'predicted_class_name' in result:
        print(f"Class Name: {result['predicted_class_name']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nClass Probabilities:")
    for i, prob in enumerate(result['probabilities']):
        class_name = result.get('predicted_class_name', f'Class {i}') if i == result['predicted_class'] else f'Class {i}'
        print(f"  {class_name}: {prob:.4f}")
    
    # Save output if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

