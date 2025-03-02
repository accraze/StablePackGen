#!/usr/bin/env python3
"""
Audio Processor Module for StablePackGen.

This module provides post-processing capabilities for audio samples,
including normalization, EQ shaping, and stereo enhancement.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Any, List, Optional, Tuple, Union
import pyloudnorm as pyln
from scipy import signal

class AudioProcessor:
    """Handles post-processing of audio samples for quality control."""
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Sample rate for audio processing (default: 44100 Hz)
        """
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(self.sample_rate)  # Initialize loudness meter
    
    def process_sample(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        processing_options: Dict[str, Any] = None
    ) -> str:
        """
        Apply post-processing to an audio sample.
        
        Args:
            input_path: Path to the input audio file
            output_path: Path to save the processed audio (if None, overwrites input)
            processing_options: Dictionary of processing options
                - normalize: Target LUFS level for normalization (e.g., -14.0)
                - eq_preset: EQ preset to apply ('kick', 'snare', 'hat', etc.)
                - stereo_width: Stereo width enhancement (0.0-2.0, 1.0 = no change)
                - high_pass: High-pass filter cutoff frequency in Hz (or None)
                - low_pass: Low-pass filter cutoff frequency in Hz (or None)
                - transient_enhance: Transient enhancement amount (0.0-1.0)
                
        Returns:
            Path to the processed audio file
        """
        # Set default output path if not provided
        if output_path is None:
            output_path = input_path
        
        # Set default processing options if not provided
        if processing_options is None:
            processing_options = {
                "normalize": -14.0,
                "eq_preset": None,
                "stereo_width": 1.0,
                "high_pass": None,
                "low_pass": None,
                "transient_enhance": 0.0
            }
        
        # Load audio file
        audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=False)
        
        # Convert mono to stereo if needed
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio])
        
        # Apply processing in the correct order
        
        # 1. High-pass filter
        if processing_options.get("high_pass"):
            audio = self._apply_high_pass(audio, processing_options["high_pass"])
        
        # 2. Low-pass filter
        if processing_options.get("low_pass"):
            audio = self._apply_low_pass(audio, processing_options["low_pass"])
        
        # 3. EQ shaping
        if processing_options.get("eq_preset"):
            audio = self._apply_eq_preset(audio, processing_options["eq_preset"])
        
        # 4. Transient enhancement
        if processing_options.get("transient_enhance", 0.0) > 0:
            audio = self._enhance_transients(audio, processing_options["transient_enhance"])
        
        # 5. Stereo width adjustment
        if processing_options.get("stereo_width", 1.0) != 1.0:
            audio = self._adjust_stereo_width(audio, processing_options["stereo_width"])
        
        # 6. Normalization (always apply as the last step)
        if processing_options.get("normalize") is not None:
            audio = self._normalize_loudness(audio, processing_options["normalize"])
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save processed audio
        sf.write(output_path, audio.T, self.sample_rate)
        
        return output_path
    
    def process_sample_pack(
        self,
        directory: str,
        processing_options: Dict[str, Dict[str, Any]] = None
    ) -> List[str]:
        """
        Apply post-processing to all samples in a directory.
        
        Args:
            directory: Directory containing audio samples
            processing_options: Dictionary mapping category names to processing options
                Example: {
                    "kicks": {"normalize": -12.0, "eq_preset": "kick"},
                    "snares": {"normalize": -14.0, "eq_preset": "snare"},
                    "default": {"normalize": -16.0}
                }
                
        Returns:
            List of paths to processed audio files
        """
        if processing_options is None:
            processing_options = {
                "default": {"normalize": -14.0}
            }
        
        processed_files = []
        
        # Walk through the directory
        for root, dirs, files in os.walk(directory):
            # Get the category from the directory name
            category = os.path.basename(root)
            
            # Skip if no audio files in this directory
            audio_files = [f for f in files if f.endswith(('.wav', '.mp3', '.aif', '.aiff'))]
            if not audio_files:
                continue
            
            print(f"Processing {len(audio_files)} samples in category: {category}")
            
            # Get processing options for this category
            category_options = processing_options.get(category, processing_options.get("default", {}))
            
            # Process each audio file
            for audio_file in audio_files:
                input_path = os.path.join(root, audio_file)
                
                try:
                    processed_path = self.process_sample(input_path, processing_options=category_options)
                    processed_files.append(processed_path)
                    print(f"  ✓ Processed: {audio_file}")
                except Exception as e:
                    print(f"  ✗ Error processing {audio_file}: {str(e)}")
        
        return processed_files
    
    def _normalize_loudness(self, audio: np.ndarray, target_lufs: float) -> np.ndarray:
        """
        Normalize audio to target LUFS loudness.
        
        Args:
            audio: Audio data (stereo)
            target_lufs: Target loudness in LUFS
            
        Returns:
            Normalized audio data
        """
        # Measure current loudness
        current_loudness = self.meter.integrated_loudness(audio.T)
        
        # Calculate gain needed
        gain_db = target_lufs - current_loudness
        
        # Apply gain
        gain_linear = 10 ** (gain_db / 20.0)
        
        # Ensure we don't clip
        peak = np.max(np.abs(audio))
        if peak * gain_linear > 1.0:
            gain_linear = 0.99 / peak
        
        return audio * gain_linear
    
    def _apply_eq_preset(self, audio: np.ndarray, preset: str) -> np.ndarray:
        """
        Apply EQ preset to audio.
        
        Args:
            audio: Audio data (stereo)
            preset: EQ preset name
            
        Returns:
            EQ'd audio data
        """
        # Define EQ presets for different sample types
        presets = {
            "kick": [
                {"type": "high_shelf", "freq": 10000, "gain": -6, "q": 0.7},
                {"type": "peaking", "freq": 100, "gain": 3, "q": 1.0},
                {"type": "high_pass", "freq": 30, "q": 0.7}
            ],
            "snare": [
                {"type": "high_shelf", "freq": 8000, "gain": 3, "q": 0.7},
                {"type": "peaking", "freq": 200, "gain": -3, "q": 1.0},
                {"type": "peaking", "freq": 3500, "gain": 4, "q": 1.5}
            ],
            "hat": [
                {"type": "high_pass", "freq": 500, "q": 0.7},
                {"type": "high_shelf", "freq": 10000, "gain": 6, "q": 0.7}
            ],
            "percussion": [
                {"type": "peaking", "freq": 400, "gain": -2, "q": 1.0},
                {"type": "peaking", "freq": 3000, "gain": 3, "q": 1.0}
            ],
            "bass": [
                {"type": "low_shelf", "freq": 100, "gain": 3, "q": 0.7},
                {"type": "high_pass", "freq": 40, "q": 0.7},
                {"type": "high_shelf", "freq": 8000, "gain": -6, "q": 0.7}
            ],
            "pad": [
                {"type": "peaking", "freq": 300, "gain": -2, "q": 1.0},
                {"type": "peaking", "freq": 3000, "gain": 2, "q": 1.0},
                {"type": "high_shelf", "freq": 10000, "gain": 3, "q": 0.7}
            ],
            "fx": [
                {"type": "peaking", "freq": 1000, "gain": 2, "q": 2.0},
                {"type": "high_shelf", "freq": 8000, "gain": 4, "q": 0.7}
            ]
        }
        
        # If preset doesn't exist, return unmodified audio
        if preset not in presets:
            return audio
        
        # Apply each filter in the preset
        processed_audio = audio.copy()
        for filter_params in presets[preset]:
            filter_type = filter_params["type"]
            freq = filter_params["freq"]
            
            if filter_type == "high_pass":
                processed_audio = self._apply_high_pass(processed_audio, freq, filter_params.get("q", 0.7))
            elif filter_type == "low_pass":
                processed_audio = self._apply_low_pass(processed_audio, freq, filter_params.get("q", 0.7))
            elif filter_type in ["peaking", "high_shelf", "low_shelf"]:
                processed_audio = self._apply_parametric_eq(
                    processed_audio, 
                    filter_type, 
                    freq, 
                    filter_params.get("gain", 0), 
                    filter_params.get("q", 1.0)
                )
        
        return processed_audio
    
    def _apply_high_pass(self, audio: np.ndarray, cutoff_freq: float, q: float = 0.7) -> np.ndarray:
        """Apply high-pass filter to audio."""
        nyquist = self.sample_rate / 2.0
        normal_cutoff = cutoff_freq / nyquist
        
        # Design filter
        b, a = signal.butter(2, normal_cutoff, btype='highpass', output='ba')
        
        # Apply to each channel
        return np.array([signal.filtfilt(b, a, channel) for channel in audio])
    
    def _apply_low_pass(self, audio: np.ndarray, cutoff_freq: float, q: float = 0.7) -> np.ndarray:
        """Apply low-pass filter to audio."""
        nyquist = self.sample_rate / 2.0
        normal_cutoff = cutoff_freq / nyquist
        
        # Design filter
        b, a = signal.butter(2, normal_cutoff, btype='lowpass', output='ba')
        
        # Apply to each channel
        return np.array([signal.filtfilt(b, a, channel) for channel in audio])
    
    def _apply_parametric_eq(
        self, 
        audio: np.ndarray, 
        filter_type: str, 
        freq: float, 
        gain_db: float, 
        q: float
    ) -> np.ndarray:
        """Apply parametric EQ to audio."""
        nyquist = self.sample_rate / 2.0
        normal_freq = freq / nyquist
        
        # Convert gain from dB to linear
        gain_linear = 10 ** (gain_db / 20.0)
        
        # Design filter based on type
        if filter_type == "peaking":
            b, a = signal.iirpeak(normal_freq, q, gain_linear)
        elif filter_type == "low_shelf":
            b, a = signal.iirfilter(
                2, normal_freq, btype='lowpass', 
                ftype='butter', output='ba'
            )
        elif filter_type == "high_shelf":
            b, a = signal.iirfilter(
                2, normal_freq, btype='highpass', 
                ftype='butter', output='ba'
            )
        else:
            return audio  # Unknown filter type
        
        # Apply to each channel
        return np.array([signal.filtfilt(b, a, channel) for channel in audio])
    
    def _adjust_stereo_width(self, audio: np.ndarray, width: float) -> np.ndarray:
        """
        Adjust stereo width of audio.
        
        Args:
            audio: Stereo audio data [2, samples]
            width: Stereo width factor (0.0 = mono, 1.0 = original, 2.0 = exaggerated)
            
        Returns:
            Width-adjusted audio
        """
        # Ensure we have stereo audio
        if audio.shape[0] != 2:
            return audio
        
        # Calculate mid and side signals
        mid = (audio[0] + audio[1]) / 2
        side = (audio[0] - audio[1]) / 2
        
        # Adjust side level based on width
        side_adjusted = side * width
        
        # Recombine to stereo
        left = mid + side_adjusted
        right = mid - side_adjusted
        
        return np.array([left, right])
    
    def _enhance_transients(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """
        Enhance transients in audio.
        
        Args:
            audio: Audio data
            amount: Enhancement amount (0.0-1.0)
            
        Returns:
            Transient-enhanced audio
        """
        # Simple transient enhancement using differentiation
        enhanced = np.zeros_like(audio)
        
        for i, channel in enumerate(audio):
            # Calculate derivative (difference)
            diff = np.diff(channel, prepend=channel[0])
            
            # Apply soft clipping to the difference signal
            diff_clipped = np.tanh(diff * 3) / 3
            
            # Mix original with enhanced version
            enhanced[i] = channel + diff_clipped * amount * 0.5
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(enhanced[i]))
            if max_val > 0.99:
                enhanced[i] = enhanced[i] / max_val * 0.99
        
        return enhanced

def get_default_processing_options() -> Dict[str, Dict[str, Any]]:
    """
    Get default processing options for different sample categories.
    
    Returns:
        Dictionary of processing options by category
    """
    return {
        "kicks": {
            "normalize": -12.0,
            "eq_preset": "kick",
            "stereo_width": 0.8,
            "high_pass": 30,
            "transient_enhance": 0.3
        },
        "snares": {
            "normalize": -14.0,
            "eq_preset": "snare",
            "stereo_width": 1.2,
            "high_pass": 100,
            "transient_enhance": 0.5
        },
        "hats": {
            "normalize": -16.0,
            "eq_preset": "hat",
            "stereo_width": 1.3,
            "high_pass": 500,
            "transient_enhance": 0.4
        },
        "percussion": {
            "normalize": -15.0,
            "eq_preset": "percussion",
            "stereo_width": 1.1,
            "high_pass": 200,
            "transient_enhance": 0.3
        },
        "basses": {
            "normalize": -13.0,
            "eq_preset": "bass",
            "stereo_width": 0.7,
            "high_pass": 40
        },
        "synths": {
            "normalize": -14.0,
            "eq_preset": "bass",
            "stereo_width": 1.1,
            "high_pass": 80
        },
        "pads": {
            "normalize": -18.0,
            "eq_preset": "pad",
            "stereo_width": 1.5
        },
        "atmospheres": {
            "normalize": -18.0,
            "eq_preset": "pad",
            "stereo_width": 1.6
        },
        "fx": {
            "normalize": -16.0,
            "eq_preset": "fx",
            "stereo_width": 1.4
        },
        "default": {
            "normalize": -14.0,
            "stereo_width": 1.0
        }
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Sample Post-Processor")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process single file command
    process_parser = subparsers.add_parser("process", help="Process a single audio file")
    process_parser.add_argument("input", help="Input audio file path")
    process_parser.add_argument("--output", help="Output audio file path")
    process_parser.add_argument("--normalize", type=float, help="Target LUFS level for normalization")
    process_parser.add_argument("--eq-preset", help="EQ preset to apply")
    process_parser.add_argument("--stereo-width", type=float, help="Stereo width factor")
    process_parser.add_argument("--high-pass", type=float, help="High-pass filter cutoff frequency")
    process_parser.add_argument("--low-pass", type=float, help="Low-pass filter cutoff frequency")
    process_parser.add_argument("--transient", type=float, help="Transient enhancement amount")
    
    # Process directory command
    batch_parser = subparsers.add_parser("batch", help="Process all audio files in a directory")
    batch_parser.add_argument("directory", help="Directory containing audio files")
    batch_parser.add_argument("--preset", choices=["default", "aggressive", "subtle"], 
                             default="default", help="Processing preset to use")
    
    args = parser.parse_args()
    
    # Create audio processor
    processor = AudioProcessor()
    
    if args.command == "process":
        # Build processing options from arguments
        options = {}
        if args.normalize is not None:
            options["normalize"] = args.normalize
        if args.eq_preset:
            options["eq_preset"] = args.eq_preset
        if args.stereo_width is not None:
            options["stereo_width"] = args.stereo_width
        if args.high_pass is not None:
            options["high_pass"] = args.high_pass
        if args.low_pass is not None:
            options["low_pass"] = args.low_pass
        if args.transient is not None:
            options["transient_enhance"] = args.transient
        
        # Process the file
        output_path = processor.process_sample(args.input, args.output, options)
        print(f"Processed audio saved to: {output_path}")
        
    elif args.command == "batch":
        # Get processing options based on preset
        if args.preset == "aggressive":
            # More extreme processing
            options = get_default_processing_options()
            for category, params in options.items():
                if "stereo_width" in params:
                    params["stereo_width"] = min(2.0, params["stereo_width"] * 1.3)
                if "transient_enhance" in params:
                    params["transient_enhance"] = min(1.0, params["transient_enhance"] * 1.5)
        elif args.preset == "subtle":
            # More subtle processing
            options = get_default_processing_options()
            for category, params in options.items():
                if "stereo_width" in params:
                    params["stereo_width"] = 0.7 + (params["stereo_width"] - 0.7) * 0.5
                if "transient_enhance" in params:
                    params["transient_enhance"] = params["transient_enhance"] * 0.5
        else:
            # Default processing
            options = get_default_processing_options()
        
        # Process the directory
        processed_files = processor.process_sample_pack(args.directory, options)
        print(f"Processed {len(processed_files)} audio files in {args.directory}")
    else:
        parser.print_help()
