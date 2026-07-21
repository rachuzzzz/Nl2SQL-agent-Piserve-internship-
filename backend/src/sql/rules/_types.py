"""
Rule-type string constants -- a zero-dependency leaf module so both
engine.py and rules/*.py can import them without a circular import.
"""

BLOCK       = "BLOCK"
WARN        = "WARN"
AUTOCORRECT = "AUTOCORRECT"
INJECT      = "INJECT"
