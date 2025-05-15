#!/bin/bash
# Script to copy the standalone repository back to original location and set up GitHub

set -e  # Exit on any error

# Define source and destination
SOURCE_DIR="/tmp/T2_XGBOOST_standalone"
DEST_DIR="/Users/macbook2024/Dropbox/AAA Backup/Transformer/T2_XGBOOST_STANDALONE"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: Source directory $SOURCE_DIR does not exist."
  exit 1
fi

# Create destination directory if it doesn't exist
echo "Creating destination directory..."
mkdir -p "$DEST_DIR"

# Copy all files except .git directory
echo "Copying repository files to $DEST_DIR..."
rsync -av --exclude='.git' "$SOURCE_DIR/" "$DEST_DIR/"

# Setup git in the new location
echo "Setting up git in the new location..."
cd "$DEST_DIR" || { echo "Failed to change directory to $DEST_DIR"; exit 1; }

# Initialize git
git init
git add .
git commit -m "Initial commit: T2 XGBOOST factor forecasting project"
git branch -M main

# Prompt for GitHub setup
echo ""
echo "===== GitHub Setup ====="
echo "Would you like to push this repository to GitHub? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
  echo "Enter your GitHub username:"
  read -r username
  
  echo "Setting up GitHub remote..."
  git remote add origin "https://github.com/$username/T2-XGBOOST.git"
  
  echo "Pushing to GitHub..."
  git push -u origin main
  
  echo "Repository successfully pushed to https://github.com/$username/T2-XGBOOST"
else
  echo "GitHub setup skipped. You can push to GitHub later with:"
  echo "  git remote add origin https://github.com/YOUR_USERNAME/T2-XGBOOST.git"
  echo "  git push -u origin main"
fi

echo ""
echo "Done! Repository copied to $DEST_DIR and git setup complete." 