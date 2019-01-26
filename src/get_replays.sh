#!/bin/bash
filter='[.[] | {map_width: .map_width, num_players: (.stats.player_statistics | length), replay: .replay, time_played: .time_played, game_id: .game_id}]'

for offset in $(seq 0 50 50000); do
  url="https://api.2018.halite.io/v1/api/user/2807/match?order_by=desc,time_played&offset=$offset&limit=50"
  wget "$url" -O games.json -q
  jq "$filter" games.json | tee --append all_replays.json
done
