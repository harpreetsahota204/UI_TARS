
import os
import logging
import json
import math
from PIL import Image
from typing import Dict, Any, List, Union, Optional
import re 
import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.utils import is_flash_attn_2_available

from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

# Constants for image resizing
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

DEFAULT_KEYPOINT_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. 
You need to perform the next action to complete the task. 

# Output Format 

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "keypoints": [
        {
            "thought": ...,
            "action": ...,
            "point_2d": <|box_start|>(x1,y1)<|box_end|>,
            }
        }
    ]
}
```
"""

DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = """You are a GUI agent specializing in classification.

Unless specifically requested for single-class output, multiple relevant classifications can be provided.

# Output format

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "classifications": [
        {
            "thought": ...,
            "label": "descriptive class label", ## Short, descriptive, and precise. Refer to your thought about the appropriate label.
        },
        {
            "thought": ...,
            "label": "descriptive class label", ## Short, descriptive, and precise. Refer to your thought about the appropriate label.
        }
    ]
}
```
"""
   
DEFAULT_OCR_SYSTEM_PROMPT = """You are a GUI agent. You specialize in text detection and recognition (OCR) in screenshots. 

You will be given a screenshot and you will need to transcribe the words in this screenshot. 

As a GUI Agent you categorize text in a screenshot into the following categories: button, link, label, checkbox, input, icon, list, or just simply "text". 
 
Read each text element only once. Do not read the same text element multiple times.

## Output Format

Always return your actions as valid JSON wrapped in ```json blocks, following this structure:


```json
{
    "text_detections": [
        {
            "thought": ..., ## Think about the category this text belongs to and if you have already read it.
            "text_category": ..., ## Refer to your thought about the appropriate category.
            "point_2d": <|box_start|>(x1,y1)<|box_end|>,
            "text": ## Transcribe text exactly as it appears
        }
    ]
}
```

Note:Do not read the same text element multiple times.
"""

DEFAULT_VQA_SYSTEM_PROMPT = "You are a GUI agent. You provide clear and concise answers to questions about any GUI. Report answers in natural language text in English."

DEFAULT_AGENTIC_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

# Action Space
click: point_2d='<|box_start|>(x1,y1)<|box_end|>'
left_double: point_2d='<|box_start|>(x1,y1)<|box_end|>'
right_single: point_2d='<|box_start|>(x1,y1)<|box_end|>' 
long_press: point_2d='<|box_start|>(x1,y1)<|box_end|>'
drag: Return a list of two points, one for the start and one for the end. point_2d='<|box_start|>(x1,y1)<|box_end|>', point_2d='<|box_start|>(x1,y1)<|box_end|>'
hotkey: key='ctrl c'
type: point_2d='<|box_start|>(x1,y1)<|box_end|>', content='xxx'
finished: point_2d='<|box_start|>(x1,y1)<|box_end|>', content='xxx'
scroll: point_2d='<|box_start|>(x1,y1)<|box_end|>', direction='down/up/right/left'
wait: point_2d='<|box_start|>(x1,y1)<|box_end|>'
open_app: point_2d='<|box_start|>(x1,y1)<|box_end|>', app_name=''
press_home: set "point_2d" to the center of the screen point_2d='<|box_start|>(x1,y1)<|box_end|>'
press_back: set "point_2d" to the center of the screen point_2d='<|box_start|>(x1,y1)<|box_end|>'

# Output Format

Thought: think about the action space, additional parameters needed, and their values 

Always return your actions as valid JSON wrapped in ```json blocks, following this structure:

```json
{
    "keypoints": [
        {
            "thought": ..., #recall your thought about the action space, additional parameters needed, and their values 
            "action": ...,
            "point_2d": <|box_start|>(x1,y1)<|box_end|>,
            "parameter_name": ... ## Refer to your thought about additional parameters.
        }
    ]
}
```

Note: Include only parameters relevant to your chosen action. Keep thoughts in English and summarize your plan with the target element in one sentence.
"""

OPERATIONS = {
    "point": DEFAULT_KEYPOINT_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT,
    "ocr": DEFAULT_OCR_SYSTEM_PROMPT,
    "agentic": DEFAULT_AGENTIC_PROMPT,
}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class UITARSModel(SamplesMixin, Model):
    """A FiftyOne model for running UITARSModel vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt  # Store custom system prompt if provided
        self._operation = operation
        self.prompt = prompt
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # Set dtype for CUDA devices
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else None
        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        model_kwargs = {
            "device_map":self.device,
            }

        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Only set specific torch_dtype for CUDA devices
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
            )

        logger.info("Loading processor")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True
        )

        self.model.eval()

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"
    
    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return self._custom_system_prompt if self._custom_system_prompt is not None else OPERATIONS[self.operation]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    def _parse_point_2d(self, point_data) -> Optional[tuple]:
        """Parse point_2d field which can be a list [x, y] or tuple (x, y)."""
        if isinstance(point_data, (list, tuple)) and len(point_data) == 2:
            try:
                x, y = point_data
                return (int(x), int(y))
            except (ValueError, TypeError):
                return None
        return None


    def _parse_json(self, s: str) -> Dict:
        """
        Parse JSON from potentially truncated model output, auto-detecting structure.
        
        Tries standard parsing first. If truncated, finds the first array key 
        in the JSON and extracts complete objects from that array.
        
        Args:
            s: Raw string containing JSON (may be in markdown blocks)
            
        Returns:
            Complete JSON if valid, otherwise dict with the detected array key
            containing any complete objects found before truncation.
        """
        # Extract JSON content from markdown code blocks if present
        if "```json" in s:
            s = s.split("```json")[1].split("```")[0].strip()  # Get content between ```json and ``` markers
        
        # Try standard JSON parsing first - this handles complete, valid JSON
        try:
            return json.loads(s)  # Return immediately if parsing succeeds
        except:
            pass  # If parsing fails, continue to recovery methods
        
        # Recovery method 1: Find the first array key pattern like "key_name": [
        array_match = re.search(r'"([^"]+)":\s*\[', s)  # Regex to find array key
        if not array_match:
            return {"items": []}  # Return empty fallback structure if no array pattern found
        
        array_key = array_match.group(1)  # Extract the actual key name from regex match
        
        # Find the exact position where array content begins
        array_pattern = f'"{array_key}": ['  # Reconstruct the exact pattern to search for
        array_start = s.find(array_pattern) + len(array_pattern)  # Calculate start position of array content
        array_content = s[array_start:]  # Extract just the array content portion
        
        # Recovery method 2: Extract complete JSON objects from the array by tracking nested braces
        objects = []  # Will hold successfully parsed objects
        depth = 0  # Track nesting level of braces
        start = -1  # Position where current object starts
        
        for i, c in enumerate(array_content):
            if c == '{':
                if depth == 0:
                    start = i  # Mark start of a new object
                depth += 1  # Increase nesting level
            elif c == '}':
                depth -= 1  # Decrease nesting level
                if depth == 0 and start >= 0:
                    try:
                        # Extract and parse complete object
                        objects.append(json.loads(array_content[start:i+1]))  # Parse individual object
                    except:
                        pass  # Skip invalid objects
        
        # Return recovered objects with original key structure
        return {array_key: objects}
    
    def _parse_and_normalize_point(self, 
                                   point_data, 
                                   image_width: int, 
                                   image_height: int
                                   ) -> Optional[tuple]:
        """Parse point and normalize for FiftyOne."""
        model_coords = self._parse_point_2d(point_data)
        if not model_coords:
            return None
        
        x, y = model_coords
        
        # Normalize to [0,1] for FiftyOne
        return (x / image_width, y / image_height)


    def _to_ocr_detections(self, parsed_output: Dict, image_width: int, image_height: int) -> fo.Keypoints:
        """Convert OCR results to FiftyOne Keypoints."""
        keypoints = []
        
        # Handle the parsed JSON structure directly
        if isinstance(parsed_output, dict):
            text_detections = parsed_output.get("text_detections", [])
        else:
            text_detections = parsed_output if isinstance(parsed_output, list) else []
        
        for detection in text_detections:
            try:
                # Parse and normalize point coordinates
                point = self._parse_and_normalize_point(
                    detection.get('point_2d'), image_width, image_height
                )
                if not point:
                    continue
                
                text = detection.get('text', '')
                text_type = detection.get('text_category', 'text')
                thought = detection.get('thought', '')
                
                if not text:
                    continue
                
                keypoint = fo.Keypoint(
                    label=str(text_type),
                    points=[list(point)],
                    text=str(text),
                    thought=thought
                )
                keypoints.append(keypoint)
                    
            except Exception as e:
                logger.debug(f"Error processing OCR detection {detection}: {e}")
                continue
                    
        return fo.Keypoints(keypoints=keypoints)

    def _to_keypoints(self, parsed_output: Dict, image_width: int, image_height: int) -> fo.Keypoints:
        """Convert keypoint detections to FiftyOne Keypoints."""
        keypoints = []
        
        # Handle the parsed JSON structure directly
        if isinstance(parsed_output, dict):
            keypoint_data = parsed_output.get("keypoints", [])
        else:
            keypoint_data = parsed_output if isinstance(parsed_output, list) else []
        
        for kp in keypoint_data:
            try:
                # Parse and normalize point coordinates
                point = self._parse_and_normalize_point(
                    kp.get('point_2d'), image_width, image_height
                )
                if not point:
                    continue
                
                action = kp.get('action', 'point')
                thought = kp.get('thought', '')
                
                keypoint = fo.Keypoint(
                    label=str(action),
                    points=[list(point)],
                    thought=thought
                )
                keypoints.append(keypoint)
            except Exception as e:
                logger.debug(f"Error processing keypoint {kp}: {e}")
                continue
                
        return fo.Keypoints(keypoints=keypoints)
    
    def _to_agentic_keypoints(self, parsed_output: Dict, image_width: int, image_height: int) -> fo.Keypoints:
        """Convert agentic actions to FiftyOne Keypoints."""
        keypoints = []
        
        # Handle the parsed JSON structure directly
        if isinstance(parsed_output, dict):
            action_data = parsed_output.get("keypoints", [])
        else:
            action_data = parsed_output if isinstance(parsed_output, list) else []
        
        for idx, kp in enumerate(action_data):
            try:
                # Parse and normalize main point coordinates
                point = self._parse_and_normalize_point(
                    kp.get('point_2d'), image_width, image_height
                )
                if not point:
                    continue
                
                action_type = kp.get("action", "unknown")
                thought = kp.get("thought", "")
                
                # Base metadata
                metadata = {
                    "sequence_idx": idx,
                    "action": action_type,
                    "thought": thought
                }
                
                # Parse additional action-specific parameters
                for key, value in kp.items():
                    if key not in ["point_2d", "action", "thought"]:
                        if key == "end_point":
                            # Parse and normalize end point coordinates for drag actions
                            end_point = self._parse_and_normalize_point(
                                value, image_width, image_height
                            )
                            if end_point:
                                metadata["end_point"] = list(end_point)
                        else:
                            metadata[key] = value
                
                keypoint = fo.Keypoint(
                    label=action_type,
                    points=[list(point)],
                    metadata=metadata
                )
                keypoints.append(keypoint)
                    
            except Exception as e:
                logger.debug(f"Error processing agentic keypoint {kp}: {e}")
                continue
                    
        return fo.Keypoints(keypoints=keypoints)

    def _to_classifications(self, parsed_output: Dict) -> fo.Classifications:
        """Convert classification results to FiftyOne Classifications."""
        classifications = []
        
        # Handle the parsed JSON structure directly
        if isinstance(parsed_output, dict):
            class_data = parsed_output.get("classifications", [])
        else:
            class_data = parsed_output if isinstance(parsed_output, list) else []
        
        for cls in class_data:
            try:
                label = cls.get("label", "unknown")
                thought = cls.get("thought", "")
                
                classification = fo.Classification(
                    label=str(label),
                    thought=thought
                )
                classifications.append(classification)
            except Exception as e:
                logger.debug(f"Error processing classification {cls}: {e}")
                continue
                
        return fo.Classifications(classifications=classifications)
        
    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Keypoints, fo.Classifications, str]:
        """Process a single image through the model and return predictions.
        
        This internal method handles the core prediction logic including:
        - Constructing the chat messages with system prompt and user query
        - Processing the image and text through the model
        - Parsing the output based on the operation type (detection/points/classification/VQA)
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            One of:
            - fo.Detections: For object detection results
            - fo.Keypoints: For keypoint detection results  
            - fo.Classifications: For classification results
            - str: For VQA text responses
            
        Raises:
            ValueError: If no prompt has been set
        """
        # Use local prompt variable instead of modifying self.prompt
        prompt = self.prompt  # Start with instance default
        
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)  # Local variable, doesn't affect instance
        
        if not prompt:
            raise ValueError("No prompt provided.")
        
        # Get original image dimensions
        original_width, original_height = image.size
        
        
        messages = [
            {
                "role": "system", 
                "content": [  
                    {
                        "type": "text",
                        "text": self.system_prompt
                    }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image", 
                        "image": sample.filepath if sample else image
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
            )
        
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text], 
            images=image_inputs,
            videos=video_inputs,
            padding=True, 
            return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=16384,
                pad_token_id=self.processor.tokenizer.eos_token_id
                )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        
        # Get image dimensions and convert to float
        input_height = float(inputs['image_grid_thw'][0][1].cpu() * 14)
        input_width = float(inputs['image_grid_thw'][0][2].cpu() * 14)
        # Convert to appropriate FiftyOne format
        if self.operation == "ocr":
            parsed_output = self._parse_json(output_text)
            return self._to_ocr_detections(parsed_output, input_width, input_height)
        elif self.operation == "point":
            parsed_output = self._parse_json(output_text)
            return self._to_keypoints(parsed_output, input_width, input_height)
        elif self.operation == "classify":
            parsed_output = self._parse_json(output_text)
            return self._to_classifications(parsed_output)
        elif self.operation == "agentic":
            parsed_output = self._parse_json(output_text)
            return self._to_agentic_keypoints(parsed_output, input_width, input_height)
        else:
            return output_text

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)