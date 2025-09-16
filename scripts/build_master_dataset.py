#!/usr/bin/env python3
"""Build a consolidated match-level dataset from StatsBomb open data."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore


def slugify(value: str) -> str:
    """Return a snake_case identifier for StatsBomb categorical names."""

    if value is None:
        return "unknown"
    lowered = value.strip().lower()
    result = []
    for char in lowered:
        if char.isalnum():
            result.append(char)
        else:
            result.append("_")
    slug = "".join(result).strip("_")
    return slug or "unknown"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class AggregatedMatch:
    match: Dict[str, Any]
    match_stats: Dict[str, Any]
    home_stats: Dict[str, Any]
    away_stats: Dict[str, Any]


class MatchAggregator:
    """Aggregate event, lineup and tracking data for a single match."""

    SHOT_ON_TARGET_OUTCOMES = {
        "saved",
        "saved_to_post",
        "goal",
        "saved_off_target",
    }

    EVENT_PREFIXES = {
        "Shot": "shots",
        "Pass": "passes",
        "Carry": "carries",
        "Dribble": "dribbles",
        "Duel": "duels",
        "Foul Committed": "fouls_committed",
        "Foul Won": "fouls_won",
        "Goal Keeper": "goalkeeper_actions",
        "Clearance": "clearances",
        "Interception": "interceptions",
        "Block": "blocks",
        "Substitution": "substitutions",
        "Ball Recovery": "ball_recoveries",
        "Ball Receipt*": "ball_receipts",
        "Pressure": "pressures",
        "Bad Behaviour": "bad_behaviours",
        "Miscontrol": "miscontrols",
        "Dispossessed": "dispossessed",
        "50/50": "fifty_fifty",
        "Error": "errors",
        "Shield": "shields",
        "Injury Stoppage": "injury_stoppages",
        "Own Goal For": "own_goals_for",
        "Own Goal Against": "own_goals_against",
        "Player On": "player_on",
        "Player Off": "player_off",
        "Half Start": "half_start",
        "Half End": "half_end",
        "Referee Ball-Drop": "referee_ball_drop",
        "Tactical Shift": "tactical_shift",
    }

    def __init__(self, match: Dict[str, Any], three_sixty: Optional[Dict[str, Any]] = None) -> None:
        self.match = match
        home_id = match["home_team"]["home_team_id"]
        away_id = match["away_team"]["away_team_id"]
        self.team_lookup = {
            home_id: "home",
            away_id: "away",
        }
        self.match_stats: Dict[str, Any] = {}
        self.match_sets: defaultdict[str, set[str]] = defaultdict(set)
        self.team_stats: Dict[str, Dict[str, Any]] = {"home": {}, "away": {}}
        self.team_sets: Dict[str, defaultdict[str, set[str]]] = {
            "home": defaultdict(set),
            "away": defaultdict(set),
        }
        self.possessions: Dict[str, set[int]] = {"home": set(), "away": set()}
        self.three_sixty = three_sixty or {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def side_for_team(self, team_id: Optional[int]) -> Optional[str]:
        if team_id is None:
            return None
        return self.team_lookup.get(team_id)

    def _target(self, side: Optional[str]) -> Dict[str, Any]:
        return self.match_stats if side is None else self.team_stats[side]

    def _target_sets(self, side: Optional[str]) -> defaultdict[str, set[str]]:
        return self.match_sets if side is None else self.team_sets[side]

    def increment(self, side: Optional[str], key: str, value: float = 1.0) -> None:
        target = self._target(side)
        target[key] = float(target.get(key, 0.0)) + float(value)

    def set_value(self, side: Optional[str], key: str, value: Any) -> None:
        if value is None:
            return
        target = self._target(side)
        if key not in target:
            target[key] = value

    def add_to_set(self, side: Optional[str], key: str, value: Optional[str]) -> None:
        if not value:
            return
        target = self._target_sets(side)
        target[key].add(value)

    def update_named_and_flag_counts(
        self,
        side: Optional[str],
        prefix: str,
        data: Dict[str, Any],
        skip_keys: Optional[Iterable[str]] = None,
    ) -> None:
        skip = set(skip_keys or [])
        for key, value in data.items():
            if key in skip or value is None:
                continue
            key_slug = slugify(str(key))
            if isinstance(value, dict):
                name = value.get("name")
                if name:
                    self.increment(side, f"{prefix}__{key_slug}__{slugify(name)}")
            elif isinstance(value, bool):
                if value:
                    self.increment(side, f"{prefix}__flag__{key_slug}")
            elif isinstance(value, (int, float)):
                self.increment(side, f"{prefix}__{key_slug}_sum", float(value))
                self.increment(side, f"{prefix}__{key_slug}_count")
            elif isinstance(value, str):
                self.increment(side, f"{prefix}__{key_slug}__{slugify(value)}")

    def accumulate_location(self, side: Optional[str], event_slug: str, location: Optional[List[float]]) -> None:
        if side is None or not isinstance(location, list) or len(location) < 2:
            return
        x, y = location[0], location[1]
        self.increment(side, f"locations__{event_slug}__x_sum", x)
        self.increment(side, f"locations__{event_slug}__y_sum", y)
        self.increment(side, f"locations__{event_slug}__count")
        if len(location) > 2 and isinstance(location[2], (int, float)):
            self.increment(side, f"locations__{event_slug}__z_sum", location[2])

    def process_three_sixty_frame(
        self,
        side: Optional[str],
        event_slug: str,
        frame: Dict[str, Any],
    ) -> None:
        self.increment(None, "three_sixty__frames_total")
        players = frame.get("freeze_frame") or []
        area = compute_polygon_area(frame.get("visible_area"))
        total_players = len(players)
        teammates = sum(1 for player in players if player.get("teammate"))
        opponents = total_players - teammates
        if side:
            self.increment(side, "three_sixty__frames", 1)
            self.increment(side, f"three_sixty__frames__{event_slug}")
            self.increment(side, "three_sixty__players_sum", total_players)
            self.increment(side, "three_sixty__teammates_sum", teammates)
            self.increment(side, "three_sixty__opponents_sum", opponents)
        if area is not None:
            self.increment(None, "three_sixty__visible_area_sum", area)
            self.increment(None, "three_sixty__visible_area_count")
            if side:
                self.increment(side, "three_sixty__visible_area_sum", area)
                self.increment(side, "three_sixty__visible_area_count")

    # ------------------------------------------------------------------
    # Event aggregation
    # ------------------------------------------------------------------
    def aggregate_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("type", {}).get("name", "Unknown")
        event_slug = slugify(event_type)
        team = event.get("team")
        side = self.side_for_team(team.get("id") if team else None)

        if side:
            self.increment(side, "events__total")
            self.increment(side, f"events__type__{event_slug}")
            prefix = self.EVENT_PREFIXES.get(event_type)
            if prefix:
                self.increment(side, f"{prefix}__total")
        self.increment(None, "match_events__total")
        self.increment(None, f"match_events__type__{event_slug}")

        for flag in ("under_pressure", "counterpress", "out", "off_camera"):
            if event.get(flag):
                if side:
                    self.increment(side, f"events__flag__{flag}")
                self.increment(None, f"match_events__flag__{flag}")

        pattern = event.get("play_pattern")
        if pattern and side:
            name = pattern.get("name")
            if name:
                self.increment(side, f"play_pattern__{slugify(name)}")

        location = event.get("location")
        self.accumulate_location(side, event_slug, location)

        duration = event.get("duration")
        if isinstance(duration, (int, float)) and side:
            self.increment(side, "events__duration_sum", float(duration))
            self.increment(side, f"events__duration__{event_slug}_sum", float(duration))
            self.increment(side, f"events__duration__{event_slug}_count")

        possession_team = event.get("possession_team")
        possession_id = event.get("possession")
        if possession_id is not None and possession_team:
            pos_side = self.side_for_team(possession_team.get("id"))
            if pos_side:
                self.possessions[pos_side].add(int(possession_id))

        if self.three_sixty:
            frame = self.three_sixty.get(event.get("id"))
            if frame:
                self.process_three_sixty_frame(side, event_slug, frame)

        handler = getattr(self, f"handle_{event_slug}", None)
        if callable(handler):
            handler(side, event)
        else:
            self.handle_generic(event_type, side, event)

    def handle_generic(self, event_type: str, side: Optional[str], event: Dict[str, Any]) -> None:
        if not side:
            return
        prefix = self.EVENT_PREFIXES.get(event_type)
        if not prefix:
            return
        data = event.get(event_type.lower().replace(" ", "_"))
        if isinstance(data, dict):
            self.update_named_and_flag_counts(side, prefix, data)

    # Dedicated handlers ------------------------------------------------
    def handle_shot(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        shot = event.get("shot") or {}
        self.increment(side, "shots__total")
        self.update_named_and_flag_counts(
            side,
            "shots",
            shot,
            skip_keys={"end_location", "freeze_frame", "statsbomb_xg", "key_pass_id"},
        )

        xg = shot.get("statsbomb_xg")
        if isinstance(xg, (int, float)):
            self.increment(side, "shots__statsbomb_xg_sum", float(xg))
            self.increment(side, "shots__statsbomb_xg_count")

        outcome = shot.get("outcome", {}).get("name")
        if outcome and slugify(outcome) in self.SHOT_ON_TARGET_OUTCOMES:
            self.increment(side, "shots__on_target")
        if outcome == "Goal":
            self.increment(side, "shots__goals")

        if shot.get("key_pass_id"):
            self.increment(side, "shots__assisted_by_pass")

        end_location = shot.get("end_location")
        if isinstance(end_location, list) and len(end_location) >= 2:
            x_end, y_end = end_location[0], end_location[1]
            self.increment(side, "shots__end_location_x_sum", x_end)
            self.increment(side, "shots__end_location_y_sum", y_end)
            self.increment(side, "shots__end_location_count")
            if len(end_location) > 2 and isinstance(end_location[2], (int, float)):
                self.increment(side, "shots__end_location_z_sum", end_location[2])

        start_location = event.get("location")
        if isinstance(start_location, list) and len(start_location) >= 2:
            distance = distance_to_goal(start_location)
            angle = angle_to_goal(start_location)
            self.increment(side, "shots__distance_to_goal_sum", distance)
            self.increment(side, "shots__distance_to_goal_count")
            if angle is not None:
                self.increment(side, "shots__angle_to_goal_sum", angle)
                self.increment(side, "shots__angle_to_goal_count")

        freeze_frame = shot.get("freeze_frame")
        if freeze_frame and not self.three_sixty.get(event.get("id")):
            frame_payload = {"freeze_frame": freeze_frame, "visible_area": shot.get("visible_area")}
            self.process_three_sixty_frame(side, "shot", frame_payload)

    def handle_pass(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        pass_data = event.get("pass") or {}
        self.increment(side, "passes__total")
        self.update_named_and_flag_counts(
            side,
            "passes",
            pass_data,
            skip_keys={"end_location", "recipient", "length", "angle"},
        )

        outcome = pass_data.get("outcome")
        if outcome is None:
            self.increment(side, "passes__completed")
        end_location = pass_data.get("end_location")
        if isinstance(end_location, list) and len(end_location) >= 2:
            x_end, y_end = end_location[0], end_location[1]
            self.increment(side, "passes__end_location_x_sum", x_end)
            self.increment(side, "passes__end_location_y_sum", y_end)
            self.increment(side, "passes__end_location_count")
            if len(end_location) > 2 and isinstance(end_location[2], (int, float)):
                self.increment(side, "passes__end_location_z_sum", end_location[2])

        length = pass_data.get("length")
        if isinstance(length, (int, float)):
            self.increment(side, "passes__length_sum", length)
            self.increment(side, "passes__length_count")
        angle = pass_data.get("angle")
        if isinstance(angle, (int, float)):
            self.increment(side, "passes__angle_sum", angle)
            self.increment(side, "passes__angle_count")

    def handle_carry(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        carry = event.get("carry") or {}
        self.increment(side, "carries__total")
        end_location = carry.get("end_location")
        if isinstance(end_location, list) and len(end_location) >= 2:
            self.increment(side, "carries__end_location_x_sum", end_location[0])
            self.increment(side, "carries__end_location_y_sum", end_location[1])
            self.increment(side, "carries__end_location_count")
        for key in ("length",):
            value = carry.get(key)
            if isinstance(value, (int, float)):
                self.increment(side, f"carries__{key}_sum", value)
                self.increment(side, f"carries__{key}_count")

    def handle_dribble(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        dribble = event.get("dribble") or {}
        self.increment(side, "dribbles__total")
        self.update_named_and_flag_counts(side, "dribbles", dribble)

    def handle_duel(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        duel = event.get("duel") or {}
        self.increment(side, "duels__total")
        self.update_named_and_flag_counts(side, "duels", duel)

    def handle_foul_committed(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        foul = event.get("foul_committed") or {}
        self.increment(side, "fouls_committed__total")
        self.update_named_and_flag_counts(side, "fouls_committed", foul)

    def handle_foul_won(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        foul = event.get("foul_won") or {}
        self.increment(side, "fouls_won__total")
        self.update_named_and_flag_counts(side, "fouls_won", foul)

    def handle_goal_keeper(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        keeper = event.get("goalkeeper") or {}
        self.increment(side, "goalkeeper_actions__total")
        self.update_named_and_flag_counts(
            side,
            "goalkeeper_actions",
            keeper,
            skip_keys={"end_location"},
        )

    def handle_clearance(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        clearance = event.get("clearance") or {}
        self.increment(side, "clearances__total")
        self.update_named_and_flag_counts(side, "clearances", clearance)

    def handle_block(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        block = event.get("block") or {}
        self.increment(side, "blocks__total")
        self.update_named_and_flag_counts(side, "blocks", block)

    def handle_interception(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        interception = event.get("interception") or {}
        self.increment(side, "interceptions__total")
        self.update_named_and_flag_counts(side, "interceptions", interception)

    def handle_substitution(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        substitution = event.get("substitution") or {}
        self.increment(side, "substitutions__total")
        self.update_named_and_flag_counts(side, "substitutions", substitution, skip_keys={"replacement"})
        replacement = substitution.get("replacement") or {}
        replacement_name = replacement.get("name")
        player = event.get("player") or {}
        player_name = player.get("name")
        if replacement_name:
            self.add_to_set(side, "substitutions__players_in", replacement_name)
        if player_name:
            self.add_to_set(side, "substitutions__players_out", player_name)

    def handle_pressure(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        self.increment(side, "pressures__total")

    def handle_bad_behaviour(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        behaviour = event.get("bad_behaviour") or {}
        self.increment(side, "bad_behaviours__total")
        self.update_named_and_flag_counts(side, "bad_behaviours", behaviour)

    def handle_miscontrol(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        miscontrol = event.get("miscontrol") or {}
        self.increment(side, "miscontrols__total")
        self.update_named_and_flag_counts(side, "miscontrols", miscontrol)

    def handle_dispossessed(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        self.increment(side, "dispossessed__total")

    def handle_fifty_fifty(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        fifty = event.get("50_50") or {}
        self.increment(side, "fifty_fifty__total")
        self.update_named_and_flag_counts(side, "fifty_fifty", fifty)

    def handle_ball_recovery(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        recovery = event.get("ball_recovery") or {}
        self.increment(side, "ball_recoveries__total")
        self.update_named_and_flag_counts(side, "ball_recoveries", recovery)

    def handle_ball_receipt(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        receipt = event.get("ball_receipt") or {}
        self.increment(side, "ball_receipts__total")
        self.update_named_and_flag_counts(side, "ball_receipts", receipt)

    def handle_starting_xi(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        tactics = event.get("tactics") or {}
        formation = tactics.get("formation")
        self.set_value(side, "starting_xi__formation", formation)
        lineup = tactics.get("lineup") or []
        for player in lineup:
            name = player.get("player", {}).get("name")
            if name:
                self.add_to_set(side, "starting_xi__players", name)
            position = player.get("position", {}).get("name")
            if position:
                self.increment(side, f"starting_xi__position__{slugify(position)}")
        self.increment(side, "starting_xi__player_count", len(lineup))

    def handle_own_goal_for(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        self.increment(side, "own_goals_for__total")

    def handle_own_goal_against(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        self.increment(side, "own_goals_against__total")

    def handle_player_on(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        player = event.get("player") or {}
        name = player.get("name")
        if name:
            self.add_to_set(side, "player_on__names", name)
        self.increment(side, "player_on__total")

    def handle_player_off(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        player = event.get("player") or {}
        name = player.get("name")
        if name:
            self.add_to_set(side, "player_off__names", name)
        self.increment(side, "player_off__total")

    def handle_injury_stoppage(self, side: Optional[str], event: Dict[str, Any]) -> None:
        if side is None:
            return
        injury = event.get("injury_stoppage") or {}
        self.increment(side, "injury_stoppages__total")
        self.update_named_and_flag_counts(side, "injury_stoppages", injury)

    def finalize(self) -> AggregatedMatch:
        for side in ("home", "away"):
            for key, values in self.team_sets[side].items():
                if values:
                    self.team_stats[side][key] = "|".join(sorted(values))
            self.team_stats[side]["possession__unique_count"] = len(self.possessions[side])
        for key, values in self.match_sets.items():
            if values:
                self.match_stats[key] = "|".join(sorted(values))
        return AggregatedMatch(
            match=self.match,
            match_stats=self.match_stats,
            home_stats=self.team_stats["home"],
            away_stats=self.team_stats["away"],
        )


def compute_polygon_area(coords: Optional[List[float]]) -> Optional[float]:
    if not coords:
        return None
    if len(coords) < 6:
        return None
    points = list(zip(coords[::2], coords[1::2]))
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def distance_to_goal(location: List[float]) -> float:
    x, y = location[0], location[1]
    goal = (120.0, 40.0)
    return math.dist((x, y), goal)


def angle_to_goal(location: List[float]) -> Optional[float]:
    if len(location) < 2:
        return None
    x, y = location[0], location[1]
    goal_x, goal_y = 120.0, 40.0
    return math.atan2(goal_y - y, goal_x - x)


class MasterDatasetBuilder:
    """Create the consolidated match-level dataset."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.matches_dir = data_dir / "matches"
        self.events_dir = data_dir / "events"
        self.lineups_dir = data_dir / "lineups"
        self.frames_dir = data_dir / "three-sixty"
        self.competitions_path = data_dir / "competitions.json"

    def build(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        competitions = self.load_competitions_lookup()
        rows: List[Dict[str, Any]] = []
        for idx, match in enumerate(self.iter_matches()):
            if limit is not None and idx >= limit:
                break
            row = self.process_match(match)
            comp_key = (row.get("competition_id"), row.get("season_id"))
            comp_data = competitions.get(comp_key)
            if comp_data:
                row.update(prefix_dict("competition", comp_data))
            self.apply_derived_metrics(row)
            rows.append(row)
        return rows

    def iter_matches(self) -> Iterable[Dict[str, Any]]:
        for comp_dir in sorted(self.matches_dir.iterdir()):
            if not comp_dir.is_dir():
                continue
            for season_file in sorted(comp_dir.glob("*.json")):
                data = read_json(season_file)
                for match in data:
                    yield match

    def process_match(self, match: Dict[str, Any]) -> Dict[str, Any]:
        match_id = match["match_id"]
        three_sixty = self.load_three_sixty(match_id)
        aggregator = MatchAggregator(match, three_sixty)
        events = read_json(self.events_dir / f"{match_id}.json")
        for event in events:
            aggregator.aggregate_event(event)
        aggregated = aggregator.finalize()
        row: Dict[str, Any] = {}
        row.update(extract_match_metadata(match))
        row.update(self.extract_lineup_stats(match, match_id))
        row.update(prefix_dict("home", aggregated.home_stats))
        row.update(prefix_dict("away", aggregated.away_stats))
        row.update(prefix_dict("match", aggregated.match_stats))
        return row

    def extract_lineup_stats(self, match: Dict[str, Any], match_id: int) -> Dict[str, Any]:
        lineup_file = self.lineups_dir / f"{match_id}.json"
        if not lineup_file.exists():
            return {}
        data = read_json(lineup_file)
        stats: Dict[str, Any] = {}
        home_team = match.get("home_team", {})
        away_team = match.get("away_team", {})
        team_map = {
            home_team.get("home_team_id"): "home",
            away_team.get("away_team_id"): "away",
        }
        for team_entry in data:
            side = team_map.get(team_entry.get("team_id"))
            if not side:
                continue
            lineup = team_entry.get("lineup") or []
            all_players = [player.get("player_name") for player in lineup if player.get("player_name")]
            starters = [
                player.get("player_name")
                for player in lineup
                if player.get("positions")
                and player["positions"][0].get("start_reason") == "Starting XI"
                and player.get("player_name")
            ]
            subs = [name for name in all_players if name not in starters]
            stats[f"{side}_lineup__player_count"] = len(all_players)
            stats[f"{side}_lineup__starters_count"] = len(starters)
            stats[f"{side}_lineup__bench_count"] = len(subs)
            stats[f"{side}_lineup__players"] = "|".join(all_players)
            stats[f"{side}_lineup__starters"] = "|".join(starters)
            stats[f"{side}_lineup__bench"] = "|".join(subs)
        return stats

    def load_three_sixty(self, match_id: int) -> Optional[Dict[str, Any]]:
        frame_file = self.frames_dir / f"{match_id}.json"
        if not frame_file.exists():
            return None
        frames = read_json(frame_file)
        return {frame["event_uuid"]: frame for frame in frames if frame.get("event_uuid")}

    def load_competitions_lookup(self) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
        if not self.competitions_path.exists():
            return {}
        data = read_json(self.competitions_path)
        lookup: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        for entry in data:
            comp_id = entry.get("competition_id")
            season_id = entry.get("season_id")
            if comp_id is None or season_id is None:
                continue
            context = {
                key: value
                for key, value in entry.items()
                if key not in {"competition_id", "season_id"}
            }
            lookup[(comp_id, season_id)] = context
        return lookup

    def apply_derived_metrics(self, row: Dict[str, Any]) -> None:
        home_goals = to_float(row.get("home_score"))
        away_goals = to_float(row.get("away_score"))
        row["match_total_goals"] = home_goals + away_goals
        row["goal_difference"] = home_goals - away_goals
        home_xg = to_float(row.get("home_shots__statsbomb_xg_sum"))
        away_xg = to_float(row.get("away_shots__statsbomb_xg_sum"))
        row["match_total_xg"] = home_xg + away_xg
        home_events = to_float(row.get("home_events__total"))
        away_events = to_float(row.get("away_events__total"))
        row["match_total_events"] = home_events + away_events
        for side in ("home", "away"):
            passes_completed = to_float(row.get(f"{side}_passes__completed"))
            passes_total = to_float(row.get(f"{side}_passes__total"))
            row[f"{side}_passes__completion_pct"] = safe_ratio(passes_completed, passes_total)
            shots_total = to_float(row.get(f"{side}_shots__total"))
            shots_goals = to_float(row.get(f"{side}_shots__goals"))
            row[f"{side}_shots__conversion_pct"] = safe_ratio(shots_goals, shots_total)
            row[f"{side}_shots__avg_xg"] = safe_ratio(
                to_float(row.get(f"{side}_shots__statsbomb_xg_sum")), shots_total
            )


def extract_match_metadata(match: Dict[str, Any]) -> Dict[str, Any]:
    competition = match.get("competition", {})
    season = match.get("season", {})
    stage = match.get("competition_stage", {})
    stadium = match.get("stadium", {})
    referee = match.get("referee", {})

    home_team = match.get("home_team", {})
    away_team = match.get("away_team", {})

    metadata = match.get("metadata", {})

    return {
        "match_id": match.get("match_id"),
        "match_date": match.get("match_date"),
        "kick_off": match.get("kick_off"),
        "competition_id": competition.get("competition_id"),
        "competition_name": competition.get("competition_name"),
        "competition_country": competition.get("country_name"),
        "season_id": season.get("season_id"),
        "season_name": season.get("season_name"),
        "competition_stage_id": stage.get("id"),
        "competition_stage_name": stage.get("name"),
        "stadium_id": stadium.get("id"),
        "stadium_name": stadium.get("name"),
        "stadium_country": (stadium.get("country") or {}).get("name"),
        "referee_id": referee.get("id"),
        "referee_name": referee.get("name"),
        "home_team_id": home_team.get("home_team_id"),
        "home_team_name": home_team.get("home_team_name"),
        "home_team_gender": home_team.get("home_team_gender"),
        "away_team_id": away_team.get("away_team_id"),
        "away_team_name": away_team.get("away_team_name"),
        "away_team_gender": away_team.get("away_team_gender"),
        "home_score": match.get("home_score"),
        "away_score": match.get("away_score"),
        "match_status": match.get("match_status"),
        "match_status_360": match.get("match_status_360"),
        "match_week": match.get("match_week"),
        "last_updated": match.get("last_updated"),
        "last_updated_360": match.get("last_updated_360"),
        "data_version": metadata.get("data_version"),
        "shot_fidelity_version": metadata.get("shot_fidelity_version"),
        "xy_fidelity_version": metadata.get("xy_fidelity_version"),
    }


def prefix_dict(prefix: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in data.items()}


def to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def write_json_rows(rows: List[Dict[str, Any]], path: Path, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
    else:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False)


def write_primary_output(path: Path, rows: List[Dict[str, Any]], df: Optional["pd.DataFrame"]) -> None:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".csv"}:
        if pd is None or df is None:
            raise SystemExit(
                "pandas is required for parquet or CSV output. Install pandas or choose a JSON destination."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
    elif suffix == ".jsonl":
        write_json_rows(rows, path, jsonl=True)
    else:
        # Default to JSON for unknown extensions
        write_json_rows(rows, path, jsonl=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a consolidated StatsBomb master dataset.")
    parser.add_argument(
        "--data-dir",
        default=Path("data"),
        type=Path,
        help="Root directory containing the open-data JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/master_matches.parquet"),
        help="Path to write the parquet dataset.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N matches (useful for debugging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = MasterDatasetBuilder(args.data_dir)
    rows = builder.build(limit=args.limit)
    df = pd.DataFrame(rows) if pd is not None else None
    write_primary_output(args.output, rows, df)
    if args.csv:
        if pd is None or df is None:
            raise SystemExit("CSV output requested but pandas is not installed.")
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
    print(f"Wrote {len(rows)} matches to {args.output}")
    if args.csv:
        print(f"Wrote CSV copy to {args.csv}")


if __name__ == "__main__":
    main()
