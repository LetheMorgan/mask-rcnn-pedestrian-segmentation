"""Class-aware IoU tracking and directional line crossing counts."""

from collections import Counter, deque
from dataclasses import dataclass, field


def iou(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return intersection / max(area(first) + area(second) - intersection, 1e-9)


@dataclass
class Track:
    identifier: int
    label: str
    box: list
    missed: int = 0
    hits: int = 1
    side: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=40))
    counted: set = field(default_factory=set)


class Tracker:
    def __init__(self, axis="y", position=0.5, margin=5, threshold=0.3,
                 max_age=15, min_hits=3):
        self.axis = axis
        self.position = position
        self.margin = margin
        self.threshold = threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = {}
        self.next_id = 1
        self.counts = Counter()

    def update(self, detections, width, height):
        candidates = []
        for identifier, track in self.tracks.items():
            track.missed += 1
            for index, detection in enumerate(detections):
                overlap = iou(track.box, detection["box"])
                if track.label == detection["class"] and overlap >= self.threshold:
                    candidates.append((overlap, identifier, index))
        matched_tracks, matched_detections = set(), {}
        for _, identifier, index in sorted(candidates, reverse=True):
            if identifier not in matched_tracks and index not in matched_detections:
                matched_tracks.add(identifier)
                matched_detections[index] = identifier
        visible, events = [], []
        for index, detection in enumerate(detections):
            identifier = matched_detections.get(index)
            if identifier is None:
                identifier = self.next_id
                self.next_id += 1
                self.tracks[identifier] = Track(identifier, detection["class"], detection["box"])
            else:
                self.tracks[identifier].hits += 1
            track = self.tracks[identifier]
            track.box, track.missed = detection["box"], 0
            x1, y1, x2, y2 = track.box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            track.history.append(center)
            coordinate = center[0] if self.axis == "x" else center[1]
            boundary = self.position * (width if self.axis == "x" else height)
            offset = coordinate - boundary
            side = 0 if abs(offset) <= self.margin else (1 if offset > 0 else -1)
            if track.hits >= self.min_hits:
                if side and track.side and side != track.side:
                    direction = "positive" if side > 0 else "negative"
                    if direction not in track.counted:
                        track.counted.add(direction)
                        self.counts[(track.label, direction)] += 1
                        events.append({"track_id": identifier, "class": track.label,
                                       "direction": direction})
                if side:
                    track.side = side
            visible.append(track)
        self.tracks = {key: track for key, track in self.tracks.items()
                       if track.missed <= self.max_age}
        return visible, events
