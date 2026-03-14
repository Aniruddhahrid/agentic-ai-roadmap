# json_practice.py
# JSON: the universal language of APIs, agents, and data exchange.
# Every API response, every tool output, every stored memory is JSON.

import json
from pathlib import Path  # modern, cleaner alternative to os.path
                          # used in all modern Python codebases


# ============================================================
# SECTION 1: JSON basics — converting between Python and JSON
# ============================================================

# Python dict → JSON string
# This is what you do before sending data to an API or saving to a file
data = {
    "agent": "research_agent",
    "status": "complete",
    "results": ["LangChain", "CrewAI", "LangGraph"],
    "score": 95,
    "verified": True
}

# json.dumps() = "dump to string"
# indent=2 makes it human-readable instead of one long line
json_string = json.dumps(data, indent=2)
print("Python dict → JSON string:")
print(json_string)
print(type(json_string))  # <class 'str'>

# JSON string → Python dict
# This is what you do when an API returns a JSON response
back_to_dict = json.loads(json_string)
print(f"\nJSON string → Python dict:")
print(back_to_dict)
print(type(back_to_dict))  # <class 'dict'>


# ============================================================
# SECTION 2: Reading and writing files with pathlib
# ============================================================

# Path() creates a path object — works on any OS (Windows, Linux, Mac)
# pathlib is the modern way. os.path is the old way. Use pathlib.

output_dir = Path("./outputs")    # a folder called outputs in current directory
output_dir.mkdir(exist_ok=True)   # create it if it doesn't exist
                                   # exist_ok=True means don't error if it already exists

# Write JSON to a file
output_file = output_dir / "agent_result.json"
# The / operator on Path objects joins paths — cleaner than string concatenation
# output_dir / "agent_result.json" = "./outputs/agent_result.json"

output_file.write_text(json.dumps(data, indent=2))
print(f"\nWritten to: {output_file}")

# Read it back
loaded_text = output_file.read_text()
loaded_data = json.loads(loaded_text)
print(f"Read back: {loaded_data['agent']} — status: {loaded_data['status']}")


# ============================================================
# SECTION 3: Working with multiple files
# ============================================================

# Create a few sample text files to work with
sample_dir = Path("./sample_texts")
sample_dir.mkdir(exist_ok=True)

# Write 3 sample files
samples = {
    "intro.txt": "AI agents are autonomous systems that perceive and act. "
                 "They use LLMs as their reasoning engine.",
    "frameworks.txt": "LangChain LangGraph CrewAI AutoGen are the major "
                      "frameworks for building agents in Python today.",
    "future.txt": "By 2027 agents will be infrastructure. "
                  "The companies that win will own the workflow not the model."
}

for filename, content in samples.items():
    (sample_dir / filename).write_text(content)

# Read all .txt files in the folder and build a summary
# glob() finds all files matching a pattern — * is wildcard
txt_files = list(sample_dir.glob("*.txt"))

results = []
for file in txt_files:
    content = file.read_text()
    word_count = len(content.split())
    results.append({
        "file": file.name,           # just the filename, not full path
        "words": word_count,
        "preview": content   # first 50 characters
    })

# Sort by word count, highest first
results.sort(key=lambda x: x["words"], reverse=True)

print("\nFile analysis:")
for r in results:
    print(f"  {r['file']}: {r['words']} words — {r['preview']}")

# Save the analysis
analysis_file = output_dir / "file_analysis.json"
analysis_file.write_text(json.dumps(results, indent=2))
print(f"\nAnalysis saved to: {analysis_file}")