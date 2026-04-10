# Clearbox 6-1-6 Standalone

Standalone desktop copy of the 6-1-6 symbol pipeline and reasoning engine.

What it does:
- Reads the current canonical lexicon from `E:/clearbox-ai-studio-v2-main/Lexical Data/Canonical`
- Writes its own local evidence store, map reports, and receipts under `./data`
- Runs the contract reasoning engine against the local standalone evidence store
- Includes a simple desktop UI for question -> reasoning output

What it does not do:
- No BM25
- No LakeSpeak
- No writes back into the current Clearbox system

Examples:
- `python -m standalone616 info`
- `python -m standalone616 health`
- `python -m standalone616 ui`
- `python -m standalone616 query --question "describe lexicon"`
- `pythonw launch_ui.pyw`