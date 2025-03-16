#!/bin/bash

# Comprehensive script to clean up repository and push changes
cd "$(dirname "$0")"

echo "=== STEP 1: Cleaning up unused example files ==="
# Remove unused example files
git rm -f examples/BostonData.py
git rm -f examples/CIBMTRDataTransformations.py
git rm -f examples/SurvPredictExpFunc.py
git rm -f examples/SurvPredictExpFuncLifelinesData.py
git rm -f examples/SurvPredict_CIBMTR_EFS_Data.py
git rm -f examples/autogluon_with_rpy2_model.py

echo "=== STEP 2: Cleaning up temporary and system files ==="
# Remove macOS system files
find . -name ".DS_Store" -delete

# Remove any other temporary files
rm -rf autogluon_imputer_models
rm -rf autogluondocs
rm -f clean_examples.sh
rm -f git_push.sh
rm -f git_push_cleanup.sh
rm -f cleanup.sh

echo "=== STEP 3: Adding all changes to git ==="
# Add all changes including modified .gitignore and README
git add -A

echo "=== STEP 4: Committing changes ==="
git commit -m "Clean up repository - remove unused examples and update documentation"

echo "=== STEP 5: Pushing to GitHub ==="
git push origin main

echo "=== DONE: Repository cleaned and changes pushed successfully! ==="

# Remove this script
rm -f cleanup_push.sh