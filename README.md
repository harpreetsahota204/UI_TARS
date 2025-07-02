# UI-TARS FiftyOne Integration

A comprehensive integration of the UI-TARS vision-language model with FiftyOne for GUI agent development and analysis. UI-TARS is an end-to-end native GUI agent model that can perceive screenshots and perform human-like interactions through unified action modeling.

<img src-"uitars-hq.gif">

- **Multi-Modal Operations**: Support for 5 different operation modes
- **Unified Action Space**: Standardized actions across desktop, mobile, and web platforms  
- **Advanced Reasoning**: System-2 thinking with explicit thought generation
- **Precise Grounding**: State-of-the-art coordinate prediction and element localization
- **FiftyOne Integration**: Seamless dataset management and visualization

## 🛠 Installation

```bash
# Install FiftyOne
pip install fiftyone

# Register the UI-TARS model source
import fiftyone.zoo as foz
foz.register_zoo_model_source("https://github.com/harpreetsahota204/UI_TARS", overwrite=True)

# Load the model
model = foz.load_zoo_model(
    "ByteDance-Seed/UI-TARS-1.5-7B",
    # install_requirements=True, #you can pass this to make sure you have all reqs installed
    )
```

## 👨🏽‍💻 Quick Start

### Load Your Dataset

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load UI dataset
dataset = load_from_hub(
    "Voxel51/ShowUI_Web",
    max_samples=200,
    shuffle=True
)

# Launch FiftyOne App
fo.launch_app(dataset)
```
### Basic Usage

#### Operation Modes

| Mode | Description | Output Format | Use Cases |
|------|-------------|---------------|-----------|
| **`vqa`** | Visual Question Answering | Text response | UI description, analysis |
| **`ocr`** | Text detection and recognition | Keypoints with text | Text extraction, element identification |
| **`point`** | Keypoint detection | Keypoints with actions | Action planning, element targeting |
| **`classify`** | UI classification | Classifications | Platform detection, UI categorization |
| **`agentic`** | Full agent actions | Keypoints with metadata | Complete automation workflows |


```python
# Visual Question Answering
model.operation = "vqa"
model.prompt = "Describe this screenshot and what the user might be doing."
dataset.apply_model(model, label_field="vqa_results")

# OCR - Extract text and UI elements
model.operation = "ocr"
model.prompt = "Point to any buttons, icons, and input fields in this UI"
dataset.apply_model(model, label_field="ocr_results")

# Keypoint Detection
model.operation = "point"
model.prompt = "Identify clickable elements for navigation"
dataset.apply_model(model, label_field="ui_keypoints")
```

### Agent Actions

```python
# Full agentic mode with reasoning
model.operation = "agentic"
dataset.apply_model(model, prompt_field="instructions", label_field="agentic_output")

# Custom system prompts
model.system_prompt = "You are a GUI testing assistant. Focus on accessibility and usability."
```

## 🎯 Detailed Operation Examples

### Visual Question Answering (`vqa`)
Perfect for understanding UI context and user intent.

```python
model.operation = "vqa"
model.prompt = "What type of application is this and what can users do here?"
dataset.apply_model(model, label_field="ui_analysis")
```

**Output**: Natural language description of the interface

### OCR Text Detection (`ocr`)
Extracts and localizes text elements with UI categorization.

```python
model.operation = "ocr"
model.prompt = "Find all interactive text elements"
dataset.apply_model(model, label_field="text_elements")
```

**Output**: Keypoints with text content and categories (button, link, input, etc.)

### Keypoint Detection (`point`)
Identifies actionable elements and suggests interactions.

```python
model.operation = "point"
model.prompt = "Locate elements needed to complete a purchase"
dataset.apply_model(model, label_field="purchase_points")
```

**Output**: Keypoints with action types and reasoning

### Classification (`classify`)
Categorizes UI characteristics for automated analysis.

```python
model.operation = "classify"
model.prompt = "Classify the platform type and primary function"
dataset.apply_model(model, label_field="ui_categories")
```

**Output**: Multiple classification labels with confidence

### Agentic Actions (`agentic`)
Complete automation with multi-step reasoning and complex actions.

```python
model.operation = "agentic"
# Uses instructions from dataset field
dataset.apply_model(model, prompt_field="task_instructions", label_field="agent_actions")
```

**Output**: Sequence of actions with coordinates, parameters, and reasoning

### Custom System Prompts

```python
# Clear default prompt
model.system_prompt = None

# Set domain-specific prompt
model.system_prompt = """
You are a web accessibility auditor. Focus on identifying:
- ARIA labels and roles
- Keyboard navigation paths  
- Color contrast issues
- Screen reader compatibility
"""

model.operation = "point"
model.prompt = "Identify accessibility issues in this interface"
```

### Using Dataset Fields

Leverage existing dataset annotations as prompts:

```python
# Use existing instruction field
dataset.apply_model(model, prompt_field="user_instructions", label_field="responses")

# Combine with custom prompts
model.prompt = "Based on the instruction, identify the next action to take"
dataset.apply_model(model, prompt_field="context", label_field="next_actions")
```

## 🎮 Action Space Reference

UI-TARS supports a comprehensive action space for cross-platform automation:

| Action Type | Parameters | Description |
|-------------|------------|-------------|
| `click` | `point_2d` | Single click at coordinates |
| `left_double` | `point_2d` | Double-click action |
| `right_single` | `point_2d` | Right-click for context menus |
| `long_press` | `point_2d` | Long press (mobile) |
| `drag` | `start_point`, `end_point` | Drag gesture between points |
| `scroll` | `point_2d`, `direction` | Scroll in specified direction |
| `type` | `content` | Text input with escape characters |
| `hotkey` | `key` | Keyboard shortcuts (e.g., "ctrl c") |
| `wait` | - | Pause for dynamic content |

## 🛡️ Best Practices

### Coordinate Handling
- All coordinates are normalized to [0,1] range
- Automatic transformation between model and original image space
- Smart resizing maintains aspect ratios

### Prompt Engineering
- Be specific about desired actions and elements
- Use task-oriented language for agentic mode
- Leverage system prompts for domain adaptation

### Error Handling
- Model includes built-in reflection and error recovery
- JSON parsing handles malformed outputs gracefully
- Coordinate validation prevents out-of-bounds errors

## 📄 Citation

```bibtex
@article{qin2025ui,
  title={UI-TARS: Pioneering Automated GUI Interaction with Native Agents},
  author={Qin, Yujia and Ye, Yining and Fang, Junjie and Wang, Haoming and Liang, Shihao and Tian, Shizuo and Zhang, Junda and Li, Jiahao and Li, Yunxin and Huang, Shijue and others},
  journal={arXiv preprint arXiv:2501.12326},
  year={2025}
}
```

## 🔗 Resources

- [UI-TARS Model Hub](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)
