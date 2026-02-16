import importlib
import pkgutil
from pathlib import Path

# Automatically import all modules in this package to trigger TaskRegistry registration
package_dir = Path(__file__).resolve().parent

for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
    importlib.import_module(f".{module_name}", package=__name__)
