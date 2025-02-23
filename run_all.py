# run_all.py
import subprocess
import sys

scripts = [
    "1_extract_basic.py",
    "1.5_table.py", 
    "2_extract_advanced.py",
    "3_user_prompt_analysis.py",
    "4_retrieval_engine.py",
    "5_LLM_layer.py"
]

def run_scripts():
    for script in scripts:
        print(f"\n{'='*40}\nRunning {script}\n{'='*40}")
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            print(f"\n⚠️  Error in {script}! Stopping pipeline.")
            sys.exit(1)
            
    print("\n✅ All scripts completed successfully!")

if __name__ == "__main__":
    run_scripts()