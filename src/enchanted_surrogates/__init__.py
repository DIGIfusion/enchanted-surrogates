from importlib.metadata import entry_points


def load_plugins():
    """
    Load all plugins registered under the 'enchanted_surrogates' entry points.

    Returns:
        dict: {plugin_name: plugin_object} for all successfully loaded plugins.
    """

    # The entry points defined in the plugins should belong to these groups
    groups = [
        "enchanted_surrogates.parsers",
        "enchanted_surrogates.runners",
    ]

    # Collect all entry points
    eps = []
    for group in groups:
        eps.extend(entry_points(group=group))

    loaded_plugins = {}

    # Load each plugin and (if successful) store it in the dictionary
    for ep in eps:
        print(f"Loading plugin: {ep.name}")
        try:
            plugin = ep.load()
            loaded_plugins[ep.name] = plugin
        except Exception as e:
            print(f"Error loading plugin '{ep.name}': {e}")

    return loaded_plugins
