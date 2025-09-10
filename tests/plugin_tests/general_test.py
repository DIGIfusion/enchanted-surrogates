import os
from enchanted_surrogates import load_plugins
from pprint import pprint
def test_atleast_one():
    plugins = load_plugins()
    assert len(plugins) > 0
    pprint(plugins)
    
if __name__ == "__main__":
    test_atleast_one()