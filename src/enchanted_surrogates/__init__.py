from importlib.metadata import entry_points
import os

def load_plugins():
    """Load all plugins registered under the 'enchanted_surrogates' entry points."""
    eps = (list(entry_points(group="enchanted_surrogates.parsers"))
           + list(entry_points(group="enchanted_surrogates.runners")))
    return {ep.name: ep.load() for ep in eps}
