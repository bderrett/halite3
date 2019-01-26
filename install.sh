#!/bin/bash
python3.6 -m pip install --target . zstd
python3.6 -m pip install --target . networkx
python3.6 MyBot.py warmup
