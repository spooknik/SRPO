#!/bin/bash
# Convert prompts.yaml to captions.json
# Usage: bash scripts/utils/convert_prompts.sh

echo "Converting prompts.yaml to captions.json..."
python scripts/utils/convert_yaml_to_json.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Success! Your captions are ready for preprocessing."
    echo ""
    echo "Next step: Run the preprocessing script"
    echo "  bash scripts/preprocess/preprocess_wan_rl_embeddings.sh"
else
    echo ""
    echo "Conversion failed. Please check the error message above."
    exit 1
fi
