import glob
import os

REPLACEMENTS = {
    "[LOAD_TIME_PCT]": "42.21%",
    "[THRASH_REDUCTION_PCT]": "100.0%",
    "[CACHE_HIT_RATE_PCT]": "57.26%",
    "[MEM_UTIL_PCT]": "0.37%",
    "[LAUNCH_TIME_PCT]": "45.14%",
    "[NEXT_CONTEXT_PREDICTION_F1]": "0.0794", # just in case
    "[F1]": "0.7745", # "Fill in F1=0.7745 where referenced"
}

md_files = glob.glob("docs/*.md") + ["README.md", "CHANGELOG.md"]

for fp in md_files:
    if not os.path.exists(fp): continue
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()
    
    orig_content = content
    for k, v in REPLACEMENTS.items():
        content = content.replace(k, v)
    
    # Also fix the PASS/FAIL placeholders in README.md
    if fp == "README.md":
        content = content.replace("[F1_PASS_FAIL]", "[FAIL]")
        content = content.replace("[CACHE_PASS_FAIL]", "[FAIL]")
        content = content.replace("[THRASH_PASS_FAIL]", "[PASS]")
        content = content.replace("[LOAD_PASS_FAIL]", "[PASS]")
        content = content.replace("[LAUNCH_PASS_FAIL]", "[PASS]")
        content = content.replace("[STABILITY_PASS_FAIL]", "[PASS]")
        content = content.replace("[MEM_PASS_FAIL]", "[FAIL]")

    if content != orig_content:
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {fp}")

print("Done.")
