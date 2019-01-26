#!/bin/bash
jq '.[] | ((.map_width|tostring) + " " + (.num_players|tostring))' all_replays.json \
    | sort | uniq -c | tr -d '"'
