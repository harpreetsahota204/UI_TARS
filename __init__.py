import logging
import os

from huggingface_hub import snapshot_download
from fiftyone.operators import types

# Import constants from zoo.py to ensure consistency
from .zoo import OPERATIONS, UITARSModel

logger = logging.getLogger(__name__)

def download_model(model_name, model_path):
    """Downloads the model.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name, model_path, **kwargs):
    """Loads the model.

    Args:
        model_name: the name of the model to load, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which the model was
            donwloaded, as declared by the ``base_filename`` field of the
            manifest
        **kwargs: optional keyword arguments that configure how the model
            is loaded

    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    
    if not model_path or not os.path.isdir(model_path):
        raise ValueError(
            f"Invalid model_path: '{model_path}'. Please ensure the model has been downloaded "
            "using fiftyone.zoo.download_zoo_model(...)"
        )
    
    logger.info(f"Loading UI-TARS-1.5 model from {model_path}")

    # Create and return the model - operations specified at apply time
    return UITARSModel(model_path=model_path, **kwargs)


def resolve_input(model_name, ctx):
    """Defines properties to collect the model's custom parameters.

    Args:
        model_name: the name of the model
        ctx: an ExecutionContext

    Returns:
        a fiftyone.operators.types.Property
    """

    inputs = types.Object()
    
    # Operation selection
    inputs.enum(
        "operation",
        values=list(OPERATIONS.keys()),
        default=None,
        required=True,
        label="Operation",
        description="Type of task to perform with UI-TARS-1.5 model",
        view=types.AutocompleteView()
    )
    
    inputs.str(
        "system_prompt",
        default=None,
        required=False,
        label="System Prompt",
        description="Optional custom system prompt",
        view=types.AutocompleteView()
    )
    
    inputs.str(
        "prompt",
        default=None,
        required=False,
        label="Prompt",
        description="Prompt for guiding operation",
        view=types.AutocompleteView()
    )
    
    return types.Property(inputs)
