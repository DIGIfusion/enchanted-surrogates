from importlib.metadata import entry_points

def load_plugins():
    """Load all plugins registered under the 'enchanted_surrogates' entry points."""
    eps = (list(entry_points(group="enchanted_surrogates.parsers"))
           + list(entry_points(group="enchanted_surrogates.runners")))
    for ep in eps:
        print(f"Loading plugin: {ep.name}")
        try:
            ep.load()
        except Exception as e:
            print(f"Error loading plugin {ep.name}: {e}")
    return {ep.name: ep.load() for ep in eps}
