from importlib.metadata import entry_points


def load_plugins():
    eps = (list(entry_points(group="enchanted_surrogates.parsers"))
           + list(entry_points(group="enchanted_surrogates.runners")))
    return {ep.name: ep.load() for ep in eps}
