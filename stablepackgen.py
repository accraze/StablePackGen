import os
import sys
from typing import Dict, List, Tuple, Any, TypedDict, Optional
from typing_extensions import Required
import anthropic
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_anthropic import ChatAnthropic
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# Define state types
class AgentState(TypedDict):
    messages: List[Any]
    actions: List[Any]
    next_step_id: int
    completed: bool
    samples_to_generate: List[Dict[str, Any]]
    current_sample_index: int

# Constants
SYSTEM_PROMPT = """You are an audio sample generation assistant specialized in creating complete sample packs. Your role is to help create high-quality samples using Stable Audio Open.
For each sample type, you will generate appropriate prompts considering:
1. Sonic characteristics (e.g., "punchy", "warm", "crisp")
2. Genre context (e.g., "techno", "house", "ambient")
3. Technical specifications (frequency content, stereo width)
4. Processing suggestions (compression, reverb, etc.)

Sample types you can create:
1. Drums:
   - Kicks (sub-heavy, punchy, etc.)
   - Snares (acoustic, electronic, layered)
   - Hi-hats (closed, open, rides)
   - Percussion (claps, rims, toms)
2. Synths:
   - Bass sounds (sub bass, acid bass, etc.)
   - Leads (plucks, stabs, sequences)
   - Pads (atmospheric, evolving)
   - FX (risers, downlifters, impacts)

For each sample request, specify:
- Detailed prompt describing the sound
- Appropriate duration (0.5-1s for drums, 2-8s for synths/pads)
- Output filename that describes the sound

Organize samples into clear categories and ensure cohesive sound design across the pack."""

class AudioGenerator:
    def __init__(self):
        self.model_info = self.init_stable_audio()
        
    def init_stable_audio(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        return {
            "model": model.to(device),
            "config": model_config,
            "device": device
        }
    
    def generate_audio_sample(
        self,
        prompt: str,
        duration: float,
        output_path: str,
        cfg_scale: float = 7.0,
        steps: int = 100,
    ) -> str:
        """Generate an audio sample using Stable Audio Open."""
        print(f"\nDebug: Starting generation for {output_path}")
        print(f"Debug: Using device: {self.model_info['device']}")
        print(f"Debug: Model config: {self.model_info['config']}")
        
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": duration
        }]
        
        try:
            print("Debug: Generating audio with diffusion model...")
            output = generate_diffusion_cond(
                self.model_info["model"],
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                sample_size=self.model_info["config"]["sample_size"],
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=self.model_info["device"]
            )
            print("Debug: Audio generation successful")
            print(f"Debug: Raw output shape: {output.shape}")
            print(f"Debug: Raw output type: {output.dtype}")
            print(f"Debug: Raw output min/max: {output.min()}, {output.max()}")
            
            print("Debug: Processing output tensor...")
            output = rearrange(output, "b d n -> d (b n)")
            print(f"Debug: After rearrange shape: {output.shape}")
            
            # Check for NaN or infinite values
            if torch.isnan(output).any():
                print("Warning: NaN values detected in output!")
            if torch.isinf(output).any():
                print("Warning: Infinite values detected in output!")
            
            output = (
                output.to(torch.float32)
                .div(torch.max(torch.abs(output)))
                .clamp(-1, 1)
                .mul(32767)
                .to(torch.int16)
                .cpu()
            )
            print(f"Debug: Final output tensor shape: {output.shape}")
            print(f"Debug: Final output tensor type: {output.dtype}")
            print(f"Debug: Final output min/max: {output.min()}, {output.max()}")
            
            # Test write to a simple file first
            test_path = "test_output.wav"
            print(f"Debug: Testing audio save with {test_path}")
            try:
                torchaudio.save(test_path, output, self.model_info["config"]["sample_rate"])
                if os.path.exists(test_path):
                    print(f"Debug: Test file successfully created, size: {os.path.getsize(test_path)} bytes")
                    os.remove(test_path)  # Clean up test file
                else:
                    print("Error: Test file was not created!")
            except Exception as e:
                print(f"Error in test save: {str(e)}")
            
            # Now try the actual save
            directory = os.path.dirname(os.path.abspath(output_path))
            print(f"Debug: Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
            
            print(f"Debug: Saving audio to {output_path}")
            print(f"Debug: Sample rate: {self.model_info['config']['sample_rate']}")
            
            try:
                torchaudio.save(output_path, output, self.model_info["config"]["sample_rate"])
                
                if os.path.exists(output_path):
                    print(f"Debug: Successfully saved file at {output_path}")
                    print(f"Debug: File size: {os.path.getsize(output_path)} bytes")
                else:
                    print(f"Error: File {output_path} was not created!")
            except Exception as e:
                print(f"Error saving audio file: {str(e)}")
                print(f"Error type: {type(e)}")
                raise
                
            # Now check duration and trim if needed
            print(f"Debug: Checking if trim needed for target duration: {duration}s")
            audio_data, sr = librosa.load(output_path, sr=self.model_info["config"]["sample_rate"])
            current_duration = len(audio_data) / sr
            
            if abs(current_duration - duration) > 0.01:  # If more than 10ms difference
                print(f"Debug: Trimming from {current_duration}s to {duration}s")
                # Calculate number of samples for target duration
                target_samples = int(duration * sr)
                if target_samples < len(audio_data):
                    # Trim to exact length
                    trimmed_audio = audio_data[:target_samples]
                else:
                    # Pad with silence if too short (unlikely with Stable Audio)
                    padding = np.zeros(target_samples - len(audio_data))
                    trimmed_audio = np.concatenate([audio_data, padding])
                
                # Save the trimmed audio
                sf.write(output_path, trimmed_audio, sr)
                print(f"Debug: Trimmed audio saved to {output_path}")
            else:
                print(f"Debug: No trimming needed, duration is already {current_duration}s")
                
            return f"Generated audio saved to {output_path} with duration {duration}s"
            
        except Exception as e:
            print(f"Error during audio generation: {str(e)}")
            print(f"Error type: {type(e)}")
            raise

from genre_templates import get_template_for_genre, get_available_genres

def generate_sample_pack_plan(genre: str = "techno") -> List[Dict[str, Any]]:
    """
    Generate a plan for a complete sample pack.
    
    Args:
        genre: The genre template to use. Available genres can be found with get_available_genres()
        
    Returns:
        List of sample definitions
    """
    try:
        # Get the sample definitions from the genre template
        sample_definitions = get_template_for_genre(genre)
        print(f"Using {genre} template with {len(sample_definitions)} samples")
        return sample_definitions
    except ValueError as e:
        # If the genre isn't supported, fall back to techno
        print(f"Warning: {str(e)}. Falling back to techno template.")
        return get_template_for_genre("techno")

class WorkflowManager:
    def __init__(self):
        self.audio_gen = AudioGenerator()
        self.llm = self._setup_llm()
        self.tools_node = self._setup_tools()
    
    def _setup_llm(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Please set it with: export ANTHROPIC_API_KEY='your-key-here'"
            )
        
        return ChatAnthropic(
            api_key=api_key,
            model="claude-3-sonnet-20240229",
            temperature=0.7,
            max_tokens=1024,
            anthropic_api_key=api_key
        )
    
    def _setup_tools(self):
        generate_tool = Tool(
            name="generate_audio",
            description="Generate an audio sample with the given parameters",
            func=self.audio_gen.generate_audio_sample
        )
        return ToolNode(tools=[generate_tool])
    
    def should_continue(self, state: Dict) -> bool:
        """Determine if we should continue processing or end."""
        return not state["completed"] and state["current_sample_index"] < len(state["samples_to_generate"])
    
    def agent_step(self, state: Dict) -> Dict:
        """Process the current state and decide the next action."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = self.llm.invoke(messages)
        
        # Update messages
        new_messages = state["messages"] + [response]
        
        # Get current sample definition
        current_sample = state["samples_to_generate"][state["current_sample_index"]]
        
        # Create action for current sample
        action = AgentAction(
            tool="generate_audio",
            tool_input={
                "prompt": current_sample["prompt"],
                "duration": current_sample["duration"],
                "output_path": current_sample["output_path"],
                "cfg_scale": 7.0,
                "steps": 100
            },
            log=""
        )
        
        # Update state
        return {
            "messages": new_messages,
            "actions": state["actions"] + [action],
            "next_step_id": state["next_step_id"] + 1,
            "completed": state["current_sample_index"] >= len(state["samples_to_generate"]) - 1,
            "samples_to_generate": state["samples_to_generate"],
            "current_sample_index": state["current_sample_index"] + 1
        }
    
    def tool_step(self, state: Dict) -> Dict:
        """Execute the tool specified in the last action."""
        last_action = state["actions"][-1]
        
        # Direct tool execution instead of using ToolNode
        if last_action.tool == "generate_audio":
            # Directly call the function with unpacked parameters
            result = self.audio_gen.generate_audio_sample(**last_action.tool_input)
        else:
            result = f"Unknown tool: {last_action.tool}"
        
        # Get current sample info
        current_sample = state["samples_to_generate"][state["current_sample_index"] - 1]
        print(f"\nGenerated {current_sample['category']}: {os.path.abspath(current_sample['output_path'])}")
        
        return {
            "messages": state["messages"] + [AIMessage(content=str(result))],
            "actions": state["actions"],
            "next_step_id": state["next_step_id"],
            "completed": state["completed"],
            "samples_to_generate": state["samples_to_generate"],
            "current_sample_index": state["current_sample_index"]
        }
    
    def create_workflow(self):
        """Create the workflow graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self.agent_step)
        workflow.add_node("tool", self.tool_step)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                True: "tool",
                False: END
            }
        )
        
        workflow.add_conditional_edges(
            "tool",
            self.should_continue,
            {
                True: "agent",
                False: END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        return workflow.compile()

def run_agent_with_prompt(prompt: str, genre: str = "techno"):
    """Run the agent with a specific prompt."""
    manager = WorkflowManager()
    workflow = manager.create_workflow()
    
    # Get sample pack plan
    samples_to_generate = generate_sample_pack_plan(genre)
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "actions": [],
        "next_step_id": 1,
        "completed": False,
        "samples_to_generate": samples_to_generate,
        "current_sample_index": 0
    }
    
    # Configure workflow with higher recursion limit
    config = {"recursion_limit": 50}  # Increased from default 25
    final_state = workflow.invoke(initial_state, config=config)
    return final_state

if __name__ == "__main__":
    # Get available genres
    available_genres = get_available_genres()
    
    # Let user select a genre
    print("\nStablePackGen - Audio Sample Pack Generator")
    print("========================================")
    print("\nAvailable genres:")
    for i, genre in enumerate(available_genres):
        print(f"{i+1}. {genre.title()}")
    
    # Get user selection
    while True:
        try:
            choice = input(f"\nSelect a genre (1-{len(available_genres)}) or enter genre name: ")
            
            # Try to parse as number
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_genres):
                    selected_genre = available_genres[choice_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(available_genres)}")
            except ValueError:
                # Try as a direct genre name
                if choice.lower() in available_genres:
                    selected_genre = choice.lower()
                    break
                else:
                    print(f"Unknown genre: {choice}. Available genres: {', '.join(available_genres)}")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
    
    print(f"\nGenerating complete {selected_genre} sample pack...")
    result = run_agent_with_prompt(f"Generate a complete {selected_genre} sample pack", genre=selected_genre)
    
    print("\nSample pack generation complete!")
    print(f"\nSamples in {selected_genre} pack by category:")
    
    # Group samples by category
    samples_by_category = {}
    for action in result["actions"]:
        path = action.tool_input["output_path"]
        # Get the relevant parts of the path
        path_parts = path.split(os.sep)
        # The category should be the last directory before the filename
        category = path_parts[-2]
        
        if category not in samples_by_category:
            samples_by_category[category] = []
        samples_by_category[category].append(os.path.basename(path))
    
    # Print organized summary
    for category, samples in samples_by_category.items():
        print(f"\n{category.upper()}:")
        for sample in samples:
            print(f"- {sample}")