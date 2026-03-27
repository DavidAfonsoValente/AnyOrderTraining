import os, sys, json
# Add aomt to path
sys.path.insert(0, os.path.join(os.getcwd(), 'aomt'))
from data.utils import load_robust_dataset

print("--- Robust Dataset Loading ---")
try:
    dataset = load_robust_dataset()
    for split in ["train", "test"]:
        print(f"\nSplit: {split} | Size: {len(dataset[split])}")
        print(f"Columns: {dataset[split].column_names}")
        
        # Sample first few
        for i in range(min(5, len(dataset[split]))):
            ex = dataset[split][i]
            conv = ex.get("conversations", [])
            text = conv[0]["value"] if conv else ""
            print(f"  [{i}] id={ex.get('id', 'N/A')} | convs={len(conv)} | start={text[:50]!r}")
            
except Exception as e:
    print(f"Robust loading failed: {e}")

print("\n--- Environment Imports ---")
try:
    import alfworld
    print("alfworld: OK")
except Exception as e:
    print(f"alfworld: missing")

try:
    import scienceworld
    print("scienceworld: OK")
except Exception as e:
    print(f"scienceworld: missing")

try:
    import web_agent_site
    print("webshop (web_agent_site): OK")
except Exception as e:
    print(f"webshop (web_agent_site): missing")
