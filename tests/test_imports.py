"""

(Added Tuesday Apr 29th, 2025)
# This script lists all modules in a package and its subpackages.
Created to assist in debugging this headache of a broken imports error I've been troubleshooting for 2-3 days.



* [TODO] -- Address the very high likelihood of name conflicts from the retrospectively poorly chosen sub-package names `plantclef.torch` and `plantclef.faiss`

"""

import pkgutil
import importlib


def list_module_tree(package_name):
    """Lists modules in a package and its subpackages."""
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        print(f"Error: Package '{package_name}' not found.")
        return

    def _recurse_modules(package, indent=0):
        for _, modname, ispkg in pkgutil.walk_packages(package.__path__):
            print("  " * indent + modname)
            if ispkg:
                try:
                    subpackage = importlib.import_module(
                        f"{package.__name__}.{modname}"
                    )
                    _recurse_modules(subpackage, indent + 1)
                except ImportError:
                    print(
                        "  " * (indent + 1)
                        + f"Error: Could not import {package.__name__}.{modname}"
                    )

    print(package_name)
    _recurse_modules(package, indent=1)


list_module_tree("plantclef")
