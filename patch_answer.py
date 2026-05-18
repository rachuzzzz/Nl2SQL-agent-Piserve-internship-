#!/usr/bin/env python3
"""
Run from the project root (where fastapi_app.py lives):
  python3 patch_answer.py

Shortens the answer text when result_data rows are present,
so the frontend shows a clean summary + table instead of
the full numbered list AND a table.
"""
import os, sys, ast

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastapi_app.py")

if not os.path.exists(TARGET):
    print(f"ERROR: fastapi_app.py not found at {TARGET}")
    sys.exit(1)

with open(TARGET) as f:
    src = f.read()

try:
    ast.parse(src)
except SyntaxError as e:
    print(f"ERROR: existing file has syntax error: {e}")
    sys.exit(1)

PATCH_MARKER = "# SHORT_ANSWER_PATCH"

if PATCH_MARKER in src:
    print("✓ Patch already applied.")
    sys.exit(0)

# Insert after _row_count is set, before the return statement
SHORT_ANSWER_CODE = f"""
        {PATCH_MARKER}
        # If we have structured rows, shorten the answer to avoid duplication
        display_answer = result.answer
        if _result_rows:
            # Extract just the first line e.g. "Found 292 result(s)..."
            first_line = result.answer.split("\\n")[0].strip()
            # If it looks like a count line, use it; otherwise generic
            if first_line.lower().startswith("found") or first_line.lower().startswith("result"):
                display_answer = first_line
            else:
                display_answer = f"Showing {{_row_count}} result{{'' if _row_count == 1 else 's'}}."
"""

# Insert just before the return statement
src = src.replace(
    "        return ChatResponse(\n"
    "            answer=result.answer,",
    SHORT_ANSWER_CODE +
    "        return ChatResponse(\n"
    "            answer=display_answer,",
    1
)

# Also update the result_data return to use display_answer context
try:
    ast.parse(src)
except SyntaxError as e:
    print(f"ERROR: patch introduced syntax error: {e}")
    sys.exit(1)

with open(TARGET, "w") as f:
    f.write(src)

print("✓ Answer shortening patch applied.")
print("  Uvicorn will auto-reload.")
