#!/bin/bash
ids="$(jq -s '.[] | .game_id' all_replays.json)"
for id in $ids; do
	 wget -nc "https://api.2018.halite.io/v1/api/user/0/match/$id/replay" -O $id.hlt
done
