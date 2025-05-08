#!/usr/bin/env bash
set -e  # exit on any error

# Save original working directory
ORIG_PWD="$(pwd)"

# Go to project directory
PROJECT_DIR="/Users/edwinhuras/Desktop/USRA_2025/GitRepo/USRA2025"
cd "$PROJECT_DIR"

# Activate virtualenv if not already
ACTIVATED=false
if [ -z "$VIRTUAL_ENV" ]; then
  if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    ACTIVATED=true
  else
    echo "⚠️  No .venv found in $PROJECT_DIR"
  fi
fi

# Merge and build
BIB_DIR="/Users/edwinhuras/Desktop/USRA_2025/GitRepo/USRA2025/Bibliography"
cd "$BIB_DIR"
echo "Merging annot.bib…"
bibtex-tools combine -o merged.bib edit_annot.bib annot.bib

echo "Building LaTeX document…"
latexmk -pdf -silent main.tex
#pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
#bibtex main.tex > /dev/null
#pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
#pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null

echo "✅ main.pdf updated."

# Return to original directory
cd "$ORIG_PWD"
