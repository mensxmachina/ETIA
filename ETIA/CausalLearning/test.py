# test_resources.py

import importlib.resources

package = 'ETIA.CausalLearning.algorithms.jar_files'

try:
    jar_files = list(importlib.resources.files(package).glob('*.jar'))
    print(f"JAR Files in '{package}':", jar_files)
except Exception as e:
    print(f"Error accessing package '{package}': {e}")
