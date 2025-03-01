"""
Genre Templates Module for StablePackGen.
Contains definitions for different music genre sample pack templates.
"""
import os
from typing import List, Dict, Any

def create_template(genre: str, prompts: Dict[str, List[Dict[str, str]]], durations: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Create a genre template from prompts and durations.
    
    Args:
        genre: The genre name
        prompts: Dictionary of category -> list of prompt dicts with "name" and "prompt" keys
        durations: Dictionary of category -> default duration for that category
        
    Returns:
        List of sample definitions
    """
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, genre.lower().replace(" ", "_"))
    
    sample_definitions = []
    
    for category, items in prompts.items():
        category_path = os.path.join(output_dir, category)
        
        for item in items:
            filename = f"{item['name']}.wav"
            sample_definitions.append({
                "prompt": item["prompt"],
                "duration": durations.get(category, 1.0),
                "output_path": os.path.join(category_path, filename),
                "category": category
            })
    
    return sample_definitions

def techno_template() -> List[Dict[str, Any]]:
    """Generate a plan for a techno sample pack."""
    prompts = {
        "kicks": [
            {
                "name": "kick_sub_punch",
                "prompt": "Deep, punchy techno kick drum with strong sub frequencies and tight attack"
            },
            {
                "name": "kick_hard_dist",
                "prompt": "Hard, distorted kick drum with mid-range presence and aggressive character"
            },
            {
                "name": "kick_dull_808",
                "prompt": "Dull 808-style kick with heavy processing for techno context" 
            }
        ],
        "snares": [
            {
                "name": "snare_snap",
                "prompt": "Snappy electronic snare with crisp transient and medium decay"
            },
            {
                "name": "snare_room",
                "prompt": "Processed acoustic snare with room reverb and parallel compression"
            }
        ],
        "hats": [
            {
                "name": "hat_closed",
                "prompt": "Tight closed hi-hat with metallic character"
            },
            {
                "name": "hat_open",
                "prompt": "Open hi-hat with long decay and bright character"
            }
        ],
        "percussion": [
            {
                "name": "clap_room",
                "prompt": "Crisp clap with medium room reverb"
            },
            {
                "name": "rim_metal",
                "prompt": "Metallic rim shot with short decay"
            }
        ],
        "synths": [
            {
                "name": "bass_sub",
                "prompt": "Deep sub bass with slight saturation and movement"
            },
            {
                "name": "bass_acid",
                "prompt": "Acid-style bass sequence with resonant filter movement"
            }
        ],
        "pads": [
            {
                "name": "pad_warm",
                "prompt": "Warm atmospheric pad with subtle modulation"
            },
            {
                "name": "pad_dark",
                "prompt": "Dark evolving texture with granular character"
            }
        ],
        "fx": [
            {
                "name": "riser_metal",
                "prompt": "Rising tension build with metallic resonance"
            },
            {
                "name": "impact_reverb",
                "prompt": "Impact hit with long reverb tail"
            }
        ]
    }
    
    durations = {
        "kicks": 1.0,
        "snares": 1.0,
        "hats": 0.5,
        "percussion": 1.0,
        "synths": 4.0,
        "pads": 8.0,
        "fx": 4.0
    }
    
    return create_template("techno", prompts, durations)

def house_template() -> List[Dict[str, Any]]:
    """Generate a plan for a house sample pack."""
    prompts = {
        "kicks": [
            {
                "name": "kick_deep",
                "prompt": "Deep, round house kick drum with smooth attack and good body"
            },
            {
                "name": "kick_classic",
                "prompt": "Classic house kick with slight compression and punch"
            },
            {
                "name": "kick_tight",
                "prompt": "Tight, punchy house kick with clean transient" 
            }
        ],
        "snares": [
            {
                "name": "snare_classic",
                "prompt": "Classic house snare with slight reverb"
            },
            {
                "name": "snare_clap_layered",
                "prompt": "Layered snare and clap with room ambience"
            }
        ],
        "hats": [
            {
                "name": "hat_tight",
                "prompt": "Tight closed hi-hat with crisp attack"
            },
            {
                "name": "hat_open_bright",
                "prompt": "Open hi-hat with bright character and medium decay"
            },
            {
                "name": "hat_pedal",
                "prompt": "Pedal hi-hat with soft character"
            }
        ],
        "percussion": [
            {
                "name": "clap_bright",
                "prompt": "Bright clap with short reverb"
            },
            {
                "name": "shaker_groove",
                "prompt": "Smooth shaker with groove feel"
            },
            {
                "name": "conga_hit",
                "prompt": "Tight conga hit with slight EQ boost"
            }
        ],
        "synths": [
            {
                "name": "bass_house",
                "prompt": "Classic house bass with filter movement"
            },
            {
                "name": "bass_deep",
                "prompt": "Deep house bass with warm character"
            }
        ],
        "pads": [
            {
                "name": "pad_chord",
                "prompt": "Warm chord pad with subtle movement"
            },
            {
                "name": "pad_airy",
                "prompt": "Airy pad with light filter sweep"
            }
        ],
        "fx": [
            {
                "name": "sweep_filter",
                "prompt": "Classic filter sweep effect"
            },
            {
                "name": "vocal_chop",
                "prompt": "Processed vocal chop with reverb"
            }
        ]
    }
    
    durations = {
        "kicks": 1.0,
        "snares": 1.0,
        "hats": 0.5,
        "percussion": 1.0,
        "synths": 2.0,
        "pads": 6.0,
        "fx": 2.0
    }
    
    return create_template("house", prompts, durations)

def ambient_template() -> List[Dict[str, Any]]:
    """Generate a plan for an ambient sample pack."""
    prompts = {
        "textures": [
            {
                "name": "texture_granular",
                "prompt": "Evolving granular texture with soft movement and airy character"
            },
            {
                "name": "texture_water",
                "prompt": "Water-like ambient texture with flowing movement"
            },
            {
                "name": "texture_space",
                "prompt": "Spacious ambient texture with wide stereo field"
            }
        ],
        "pads": [
            {
                "name": "pad_lush",
                "prompt": "Lush ambient pad with subtle modulation and rich harmonics"
            },
            {
                "name": "pad_evolving",
                "prompt": "Slowly evolving atmospheric pad with gentle movement"
            },
            {
                "name": "pad_dark",
                "prompt": "Dark atmospheric pad with subtle dissonance"
            }
        ],
        "drones": [
            {
                "name": "drone_deep",
                "prompt": "Deep sustained drone with subtle harmonic movement"
            },
            {
                "name": "drone_mid",
                "prompt": "Mid-range drone with slight tension and resolution"
            },
            {
                "name": "drone_high",
                "prompt": "High atmospheric drone with airy character"
            }
        ],
        "percussion": [
            {
                "name": "perc_soft",
                "prompt": "Soft percussive hit with long reverb tail"
            },
            {
                "name": "perc_bell",
                "prompt": "Bell-like percussion with resonant character"
            }
        ],
        "atmospheres": [
            {
                "name": "atmos_space",
                "prompt": "Spacious atmospheric texture with depth and movement"
            },
            {
                "name": "atmos_tonal",
                "prompt": "Tonal atmospheric background with subtle pitch movement"
            }
        ],
        "keys": [
            {
                "name": "keys_pad",
                "prompt": "Soft piano-like sound with heavy processing and reverb"
            },
            {
                "name": "keys_bell",
                "prompt": "Bell-like keyboard sound with long decay"
            }
        ]
    }
    
    durations = {
        "textures": 12.0,
        "pads": 16.0,
        "drones": 20.0,
        "percussion": 4.0,
        "atmospheres": 16.0,
        "keys": 8.0
    }
    
    return create_template("ambient", prompts, durations)

def drum_and_bass_template() -> List[Dict[str, Any]]:
    """Generate a plan for a drum and bass sample pack."""
    prompts = {
        "kicks": [
            {
                "name": "kick_sub",
                "prompt": "Deep sub kick with tight attack for drum and bass"
            },
            {
                "name": "kick_punch",
                "prompt": "Punchy kick with mid presence for drum and bass"
            }
        ],
        "snares": [
            {
                "name": "snare_tight",
                "prompt": "Tight snare with sharp attack for drum and bass"
            },
            {
                "name": "snare_processed",
                "prompt": "Heavily processed layered snare with snap and body"
            },
            {
                "name": "snare_break",
                "prompt": "Break-style snare with room character"
            }
        ],
        "hats": [
            {
                "name": "hat_closed_tight",
                "prompt": "Tight closed hat with crisp attack"
            },
            {
                "name": "hat_open_bright",
                "prompt": "Open hat with bright character and decay"
            }
        ],
        "percussion": [
            {
                "name": "perc_metallic",
                "prompt": "Metallic percussion hit with tight character"
            },
            {
                "name": "perc_processed",
                "prompt": "Heavily processed percussion with unique character"
            }
        ],
        "basses": [
            {
                "name": "bass_reese",
                "prompt": "Classic reese bass with movement and distortion"
            },
            {
                "name": "bass_sub",
                "prompt": "Clean sub bass with slight harmonics"
            },
            {
                "name": "bass_neuro",
                "prompt": "Heavily processed neuro bass with movement and distortion"
            }
        ],
        "atmospheres": [
            {
                "name": "atmos_dark",
                "prompt": "Dark atmospheric pad with tension"
            },
            {
                "name": "atmos_sci_fi",
                "prompt": "Sci-fi atmospheric texture with movement"
            }
        ],
        "fx": [
            {
                "name": "fx_riser",
                "prompt": "Tense rising effect with accelerating movement"
            },
            {
                "name": "fx_impact",
                "prompt": "Heavy impact hit with distortion and reverb"
            }
        ]
    }
    
    durations = {
        "kicks": 0.8,
        "snares": 0.8,
        "hats": 0.5,
        "percussion": 0.8,
        "basses": 4.0,
        "atmospheres": 8.0,
        "fx": 4.0
    }
    
    return create_template("drum_and_bass", prompts, durations)

# Dictionary to map genre names to their template functions
GENRE_TEMPLATES = {
    "techno": techno_template,
    "house": house_template,
    "ambient": ambient_template,
    "drum_and_bass": drum_and_bass_template
}

def get_available_genres() -> List[str]:
    """Return a list of available genre templates."""
    return list(GENRE_TEMPLATES.keys())

def get_template_for_genre(genre: str) -> List[Dict[str, Any]]:
    """Get the sample pack template for the specified genre."""
    if genre.lower() not in GENRE_TEMPLATES:
        raise ValueError(f"Unknown genre: {genre}. Available genres: {', '.join(get_available_genres())}")
    
    template_func = GENRE_TEMPLATES[genre.lower()]
    sample_definitions = template_func()
    
    # Create directories for all categories
    categories = set(sample["category"] for sample in sample_definitions)
    output_dir = os.path.dirname(sample_definitions[0]["output_path"])
    
    # Create genre directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create category directories
    for category in categories:
        category_path = os.path.join(output_dir, category)
        os.makedirs(category_path, exist_ok=True)
    
    return sample_definitions