class BasketballTracker:
    """
    Maps unstable boxmot IDs -> stable custom IDs per class.

    Stability trick: when a boxmot_id disappears, its custom slot is held in
    a cooldown for GRACE_FRAMES before being freed.  This prevents the slot
    from immediately being recycled to a different player during brief
    occlusions, which was the main cause of flickering IDs.
    """

    GRACE_FRAMES = 45   # ~1.5 s at 30 fps — tune if needed

    def __init__(self):
        self.names = {0: "basketball", 1: "net", 2: "player", 3: "referee"}

        # Classes that get custom IDs (basketball is drawn raw, never tracked)
        self.max_ids = {"player": 10, "referee": 4, "net": 2}

        # boxmot_id -> custom_id  (currently visible tracks)
        self.active = {c: {} for c in self.max_ids}

        # custom_id -> frames_remaining  (recently departed tracks)
        # Slot is NOT available for reuse until counter reaches 0.
        self.cooling = {c: {} for c in self.max_ids}

    # ------------------------------------------------------------------
    def update_ids(self, boxmot_tracks):
        """
        Call once per frame with the raw boxmot output array.
        Returns a list of dicts ready for drawing.
        """
        tracked_objects = []
        if len(boxmot_tracks) == 0:
            self._tick_cooldowns()
            return tracked_objects

        # 1. Collect which boxmot IDs are alive this frame, per class
        current_ids = {c: set() for c in self.max_ids}
        for track in boxmot_tracks:
            _, _, _, _, ori_id, _, cls_idx = track[:7]
            cls_name = self.names[int(cls_idx)]
            if cls_name in current_ids:
                current_ids[cls_name].add(int(ori_id))

        # 2. Move departed boxmot IDs into cooldown
        for cls_name, id_map in self.active.items():
            departed = set(id_map.keys()) - current_ids[cls_name]
            for dead_id in departed:
                custom_id = id_map.pop(dead_id)
                self.cooling[cls_name][custom_id] = self.GRACE_FRAMES

        # 3. Tick down cooldowns; free expired slots
        self._tick_cooldowns()

        # 4. Assign custom IDs and build output
        for track in boxmot_tracks:
            x1, y1, x2, y2, ori_id, score, cls_idx = track[:7]
            cls_name = self.names[int(cls_idx)]
            ori_id   = int(ori_id)

            custom_id = self._assign(cls_name, ori_id)

            tracked_objects.append({
                "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                "track_id":   custom_id,
                "class_name": cls_name,
                "conf":       float(score),
            })

        return tracked_objects

    # ------------------------------------------------------------------
    def _assign(self, cls_name, ori_id):
        if cls_name not in self.max_ids:
            return ori_id

        id_map  = self.active[cls_name]
        cooling = self.cooling[cls_name]
        max_id  = self.max_ids[cls_name]

        if ori_id in id_map:
            return id_map[ori_id]

        # Find the lowest custom ID that is neither active nor cooling
        occupied = set(id_map.values()) | set(cooling.keys())
        for candidate in range(1, max_id + 1):
            if candidate not in occupied:
                id_map[ori_id] = candidate
                return candidate

        # All slots occupied — use raw boxmot ID so player is still drawn
        return ori_id

    def _tick_cooldowns(self):
        for cooling in self.cooling.values():
            expired = [cid for cid, t in cooling.items() if t <= 1]
            for cid in expired:
                del cooling[cid]
            for cid in list(cooling):
                if cid not in expired:
                    cooling[cid] -= 1