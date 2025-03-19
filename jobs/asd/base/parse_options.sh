#!/usr/bin/bash

# Parse command-line options.
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --*=*)
            option_name=$(echo "$key" | sed 's/--//')
            name_=$(echo "$option_name" | cut -d'=' -f1)
            value_=$(echo "$option_name" | cut -d'=' -f2)
            export "$name_"="$value_"
            shift
            ;;
        --help)
            echo "Usage: ./base.sh --name=value"
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done
