# test_setup.py
# Purpose: verify our entire Day 1 setup is working correctly.

import sys           # built-in: gives info about the Python interpreter running this file
import requests      # installed via pip: for making HTTP calls to APIs
from dotenv import load_dotenv  # installed via pip: loads secret keys from .env files

# sys.version = full Python version string
# sys.executable = exact path of the Python binary running this script
# f-strings (f"...") = Python's way of embedding variables inside strings
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

# Confirms both packages loaded without errors
print(f"requests version: {requests.__version__}")
print("python-dotenv: loaded successfully")

print("\n✅ Day 1 complete. Environment is working.")
