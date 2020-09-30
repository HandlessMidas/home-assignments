#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def find_corners(frame_0, frame_1, poses_0, corner_ids, max_corners, last_id):
    poses_1, status, err = cv2.calcOpticalFlowPyrLK(np.uint8(frame_0 * 255), np.uint8(frame_1 * 255), poses_0,
                                                    None, winSize=(15, 15), maxLevel=2,
                                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    reverse_poses_1, reverse_status, reverse_err = cv2.calcOpticalFlowPyrLK(np.uint8(frame_1 * 255), np.uint8(frame_0 * 255), poses_1,
                                                                            None, winSize=(15, 15), maxLevel=2,
                                                                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    quality = abs(poses_0 - reverse_poses_1)
    is_good_quality = []
    for i in quality:
        if max(i[0][0], i[0][1]) < 1:
            is_good_quality.append(True)
        else:
            is_good_quality.append(False)
    corner_ids = corner_ids[is_good_quality]
    poses_0 = poses_1[is_good_quality]

    if len(poses_0) < max_corners:
        new_poses_0 = cv2.goodFeaturesToTrack(
            frame_0, maxCorners=max_corners, qualityLevel=0.01, minDistance=7)
        remainder = min(max_corners - len(poses_0), len(new_poses_0))

        dist_from_new_points = []
        for point in new_poses_0.reshape((-1, 2)):
            dist = np.sqrt(np.sum((point - poses_0) ** 2, axis=2))
            dist_from_new_points.append(dist.min())

        suitable_dist = np.sort(dist_from_new_points)[-remainder]
        new_poses_0 = new_poses_0[dist_from_new_points >= suitable_dist]

        new_corner_ids = np.array(range(last_id, last_id + remainder))
        last_id += remainder
        corner_ids = np.concatenate([corner_ids, new_corner_ids])

        poses_0 = np.concatenate([poses_0, new_poses_0])

    return poses_0, corner_ids, last_id


def corners_concat(ids, poses, compressed_ids, compressed_poses):
    concat_ids = np.concatenate([ids, compressed_ids])
    concat_corners = np.concatenate([poses, compressed_poses * 2])
    concat_radius = np.concatenate([np.array(np.full(len(poses), 14)), np.array(
        np.full(len(compressed_poses), 7))])
    corners = FrameCorners(concat_ids, concat_corners, concat_radius)
    return corners


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]

    max_corners = 1000
    poses = cv2.goodFeaturesToTrack(
        image_0, maxCorners=max_corners, qualityLevel=0.01, minDistance=7)
    corner_ids = np.array(range(len(poses)))
    last_id = len(poses)

    compressed_image_0 = np.array([i[::2] for i in image_0[::2]])

    compressed_poses = cv2.goodFeaturesToTrack(
        image_0, maxCorners=max_corners, qualityLevel=0.01, minDistance=7)
    compressed_last_id = max_corners * \
        len(frame_sequence) + len(compressed_poses) + 1
    compressed_ids = np.array(
        range(max_corners * len(frame_sequence) + 1, compressed_last_id))

    corners = corners_concat(
        corner_ids, poses, compressed_ids, compressed_poses)
    builder.set_corners_at_frame(0, corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        compressed_image_1 = np.array([i[::2] for i in image_1[::2]])

        poses, corner_ids, last_id = find_corners(
            image_0, image_1, poses, corner_ids, max_corners, last_id)

        compressed_poses, compressed_ids, compressed_last_id = find_corners(compressed_image_0, compressed_image_1,
                                                                            compressed_poses, compressed_ids,
                                                                            max_corners, compressed_last_id)

        corners = corners_concat(
            corner_ids, poses, compressed_ids, compressed_poses)
        builder.set_corners_at_frame(frame, corners)

        image_0 = image_1
        compressed_image_0 = compressed_image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
