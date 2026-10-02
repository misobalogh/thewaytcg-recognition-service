from dataclasses import dataclass
import json
from threading import Lock

import cv2
import numpy as np

from src.the_way_recognition.config import Settings


@dataclass(frozen=True)
class Reference:
    card_id: str
    name: str
    points: np.ndarray
    shape: tuple[int, int]
    descriptors: np.ndarray
    matcher: cv2.FlannBasedMatcher


@dataclass(frozen=True)
class Candidate:
    reference: Reference
    inliers: int = 0
    inlier_ratio: float = 0.0
    coverage: float = 0.0
    score: float = 0.0
    valid: bool = False


@dataclass(frozen=True)
class RecognitionResult:
    candidate: Candidate | None = None
    accepted: bool = False
    confidence: str = "none"
    # None means there is no second geometrically valid candidate to compare.
    margin: float | None = None


def resize_image(image: np.ndarray, max_dim: int) -> np.ndarray:
    scale = min(1.0, max_dim / max(image.shape[:2]))
    if scale < 1:
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def decode_image(contents: bytes) -> np.ndarray:
    if not contents:
        raise ValueError("Invalid image file: empty upload")
    try:
        image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except cv2.error as exc:
        raise ValueError("Invalid image file") from exc
    if image is None:
        raise ValueError("Invalid image file")
    return image


class SIFTRecognitionService:
    """One immutable catalog per worker; shared OpenCV indexes are serialized."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = Lock()
        cv2.setNumThreads(settings.OPENCV_THREADS)
        cv2.setRNGSeed(42)
        self._sift = cv2.SIFT_create(nfeatures=settings.SIFT_NFEATURES)
        self.references = self._load_references()
        self._global_matcher = self._matcher(settings.SIFT_GLOBAL_CHECKS)
        self._global_matcher.add([ref.descriptors for ref in self.references])
        self._global_matcher.train()

    @staticmethod
    def _matcher(checks: int) -> cv2.FlannBasedMatcher:
        return cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=checks))

    def _features(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        keypoints, descriptors = self._sift.detectAndCompute(image, None)
        points = np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)
        return points, descriptors

    def _load_references(self) -> list[Reference]:
        image_dir = self.settings.REFERENCE_IMAGE_DIR
        metadata_dir = self.settings.CARD_METADATA_DIR
        metadata_paths = sorted(metadata_dir.glob("*.json"))
        if not metadata_paths:
            raise ValueError(f"No card metadata found in {metadata_dir}")
        image_paths = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        images = {p.stem: p for p in image_paths}
        if len(images) != len(image_paths):
            raise ValueError("Reference image IDs must be unique")
        metadata_ids = {p.stem for p in metadata_paths}
        if set(images) != metadata_ids:
            missing = sorted(metadata_ids - set(images))
            extra = sorted(set(images) - metadata_ids)
            raise ValueError(f"Reference/metadata IDs differ: missing images={missing}, missing metadata={extra}")
        references = []
        for path in image_paths:
            metadata_path = metadata_dir / f"{path.stem}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            name = metadata.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Missing card name in {metadata_path}")
            image = resize_image(decode_image(path.read_bytes()), self.settings.MAX_IMAGE_DIM)
            points, descriptors = self._features(image)
            if descriptors is None or len(descriptors) < 4:
                raise ValueError(f"Not enough SIFT features in {path}")
            matcher = self._matcher(64)
            matcher.add([descriptors])
            matcher.train()
            references.append(Reference(path.stem, name.strip(), points, image.shape, descriptors, matcher))
        return references

    def _shortlist(self, descriptors: np.ndarray) -> list[Reference]:
        pairs = self._global_matcher.knnMatch(descriptors, k=2)
        votes = np.zeros(len(self.references), dtype=np.float64)
        for pair in pairs:
            if len(pair) != 2:
                continue
            best, second = pair
            weight = 0.1 + max(0.0, 1.0 - best.distance / max(second.distance, 1e-6))
            votes[best.imgIdx] += weight
        order = np.argsort(-votes, kind="stable")
        return [self.references[int(i)] for i in order[:self.settings.SIFT_SHORTLIST_SIZE]]

    def _compare(self, image: np.ndarray, points: np.ndarray, descriptors: np.ndarray, ref: Reference) -> Candidate:
        pairs = ref.matcher.knnMatch(descriptors, k=2)
        good = sorted((pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < self.settings.SIFT_RATIO * pair[1].distance), key=lambda m: m.distance)
        unique, used = [], set()
        for match in good:
            if match.trainIdx not in used:
                unique.append(match)
                used.add(match.trainIdx)
        if len(unique) < 4:
            return Candidate(ref)
        src = ref.points[[m.trainIdx for m in unique]]
        dst = points[[m.queryIdx for m in unique]]
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if homography is None or mask is None or not np.isfinite(homography).all():
            return Candidate(ref)
        keep = mask.ravel().astype(bool)
        inliers = int(keep.sum())
        height, width = ref.shape
        coverage = cv2.contourArea(cv2.convexHull(src[keep])) / (height * width) if inliers >= 3 else 0.0
        corners = cv2.perspectiveTransform(np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]).reshape(-1, 1, 2), homography)
        query_height, query_width = image.shape
        finite = bool(np.isfinite(corners).all())
        area = abs(cv2.contourArea(corners)) / (query_height * query_width) if finite else 0.0
        plausible = finite and cv2.isContourConvex(corners) and 0.01 <= area <= 1.5
        inside = finite and bool(((corners[:, 0, 0] >= -0.25 * query_width) & (corners[:, 0, 0] <= 1.25 * query_width)
                                  & (corners[:, 0, 1] >= -0.25 * query_height) & (corners[:, 0, 1] <= 1.25 * query_height)).all())
        ratio = inliers / len(unique)
        valid = (inliers >= self.settings.SIFT_MIN_INLIERS and ratio >= self.settings.SIFT_MIN_INLIER_RATIO
                 and coverage >= self.settings.SIFT_MIN_COVERAGE and plausible and inside)
        score = float(inliers * ratio * np.sqrt(coverage)) if valid else 0.0
        return Candidate(ref, inliers, float(ratio), float(coverage), score, bool(valid))

    def _select(self, ranked: list[Candidate]) -> RecognitionResult:
        if not ranked:
            return RecognitionResult()
        top = ranked[0]
        second_score = ranked[1].score if len(ranked) > 1 else 0.0
        margin = top.score / second_score if second_score > 0 else None
        accepted = top.valid and (margin is None or margin >= self.settings.SIFT_MIN_MARGIN)
        confidence = "none"
        if accepted:
            # Heuristic strength labels, not calibrated probabilities.
            strong_margin = margin is None or margin >= 2.0
            if top.inliers >= 30 and top.inlier_ratio >= 0.6 and strong_margin:
                confidence = "high"
            elif strong_margin:
                confidence = "medium"
            else:
                confidence = "low"
        return RecognitionResult(top, accepted, confidence, margin)

    def recognize(self, contents: bytes) -> RecognitionResult:
        image = resize_image(decode_image(contents), self.settings.MAX_IMAGE_DIM)
        with self._lock:
            # OpenCV's RNG and mutable matcher internals are confined to this call.
            cv2.setRNGSeed(42)
            points, descriptors = self._features(image)
            if descriptors is None or len(descriptors) < 4:
                return RecognitionResult()
            candidates = self._shortlist(descriptors)
            ranked = sorted((self._compare(image, points, descriptors, ref) for ref in candidates), key=lambda c: (c.score, c.inliers), reverse=True)
            return self._select(ranked)
