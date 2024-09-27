#!/bin/bash

# Define paths
SOURCE_DIR="../ETIA"  # Adjust this if needed
DOCS_DIR="docs"
SOURCE_DOCS_DIR="$DOCS_DIR/"
BUILD_DIR="$DOCS_DIR/_build"
PREWRITTEN_DIR = "$DOCS_DIR/prewritten_rst/"
# Generate .rst files from Python modules
sphinx-apidoc --module-first --implicit-namespaces -o $SOURCE_DOCS_DIR $SOURCE_DIR

# Function to remove the ETIA. prefix from filenames and content
remove_prefix() {
  for file in $SOURCE_DOCS_DIR/ETIA.*.rst; do
    # Strip the prefix from the filename
    new_file=$(echo $file | sed 's/ETIA.//')
    mv "$file" "$new_file"

    # Strip the prefix from the content
    sed -i 's/ETIA\.//g' "$new_file"
  done
}

# Remove the ETIA. prefix
remove_prefix

# Copy our pre-written index.rst and module description files to the appropriate locations
# Ensure the source files are in place
cp -r ./prewritten_rst/* $SOURCE_DOCS_DIR

# Build the HTML documentation
sphinx-build -E -b html $SOURCE_DOCS_DIR $BUILD_DIR

# Print the location of the generated documentation
echo "Documentation has been generated at $BUILD_DIR/index.html"
