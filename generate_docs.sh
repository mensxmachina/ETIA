#!/bin/bash

# Define paths
SOURCE_DIR="AutoCD"
DOCS_DIR="docs"
SOURCE_DOCS_DIR="$DOCS_DIR/"
BUILD_DIR="$DOCS_DIR/_build"

# Generate .rst files
sphinx-apidoc --module-first --implicit-namespaces -o $SOURCE_DOCS_DIR $SOURCE_DIR

# Function to remove the ETIA. prefix from filenames and content
remove_prefix() {
  for file in $SOURCE_DOCS_DIR/ETIA.*.rst; do
    # Strip the prefix from the filename
    new_file=$(echo $file | sed 's/ETIA.//')
    mv "$file" "$new_file"

    # Strip the prefix from the content
    sed -i '' 's/ETIA\.//g' "$new_file"
  done
}

# Remove the ETIA. prefix
remove_prefix

# Build the HTML documentation
sphinx-build -b html $SOURCE_DOCS_DIR $BUILD_DIR

# Print the location of the generated documentation
echo "Documentation has been generated at $BUILD_DIR/index.html"
