#!/bin/bash

# Script to cleanup repository and push changes to GitHub
cd "$(dirname "$0")"

# Make sure cleanup script is executable
chmod +x cleanup.sh

# Run the cleanup script
./cleanup.sh

# Add all changes and deletions
git add -A

# Commit the changes
git commit -m "Remove unnecessary files and clean up repository"

# Push to GitHub
git push origin main

echo "Repository cleaned up and changes pushed to GitHub successfully!"