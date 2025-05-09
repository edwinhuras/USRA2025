#!/usr/bin/env bash
set -e  # exit on any error

# Save original working directory
ORIG_PWD="$(pwd)"

# Go to project directory
PROJECT_DIR="/Users/edwinhuras/Desktop/USRA_2025/GitRepo/USRA2025"
cd "$PROJECT_DIR"

# Activate virtualenv if not already
if [ -z "$VIRTUAL_ENV" ] && [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# Merge and build
BIB_DIR="/Users/edwinhuras/Desktop/USRA_2025/GitRepo/USRA2025/Bibliography"
cd "$BIB_DIR"

# Delete stale .bbl
#rm -f main.bbl

echo "Merging .bib files…"
# bibtex-tools will ask for confirmation to overwrite dupllicates
# this is a workaround to automate the process and suppress the output
yes "" | bibtex-tools combine -o merged.bib edit_annot.bib annot.bib > /dev/null 2>&1

echo "Building PDF…"
latexmk -pdf -silent main.tex

echo "Build complete: main.pdf updated."

# Return to original directory
cd "$ORIG_PWD"
