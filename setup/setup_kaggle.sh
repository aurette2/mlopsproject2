#!/bin/bash

# Create the .kaggle directory if it does not exist
mkdir -p ~/.kaggle

# Move kaggle.json to the .kaggle directory
mv ~/Downloads/kaggle.json ~/.kaggle/

# Change file permissions
chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
! kaggle datasets download awsaf49/brats20-dataset-training-validation

# Un zip the data set
! un/home/jupyter/s20-dataset-training-validation.zip

echo "Kaggle setup completed successfully."

#____________________________________To handle the image of the model UNET_________________________________#
# If Homebrew is not installed on your system
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

printf '%s\n' '' 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/omer/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"



