#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="${1:-configs/sft_lora_270m.yml}"
shift || true

export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

show_help() {
    echo "Usage: ./run_experiment.sh <config_file> [options]"
    echo ""
    echo "Arguments:"
    echo "  config_file    Path to YAML config file (default: configs/sft_lora_270m.yml)"
    echo ""
    echo "Options:"
    echo "  --debug        Run in debug mode (single batch)"
    echo "  --download     Download datasets before training"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_experiment.sh configs/sft_lora_270m.yml --debug"
    echo "  ./run_experiment.sh configs/ppo_rl_270m.yml --download"
}

DOWNLOAD_DATA=false
DEBUG_MODE=false
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --help)
            show_help
            exit 0
            ;;
        --download)
            DOWNLOAD_DATA=true
            ;;
        --debug)
            DEBUG_MODE=true
            EXTRA_ARGS="$EXTRA_ARGS --debug"
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "========================================"
echo "Gemma Alignment Experiment Runner"
echo "========================================"
echo "Config: $CONFIG_FILE"
echo "Debug mode: $DEBUG_MODE"
echo ""

if [ "$DOWNLOAD_DATA" = true ]; then
    echo "Downloading datasets..."
    python data/download_datasets.py --dataset safety --splits train,val,test
    echo ""
fi

echo "Starting training..."
python -m src.experiments.runner --config "$CONFIG_FILE" $EXTRA_ARGS

echo ""
echo "========================================"
echo "Experiment completed successfully"
echo "========================================"
