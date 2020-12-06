#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import random
import cv2
import sortednp as snp

from corners import CornerStorage
from _corners import filter_frame_corners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    TriangulationParameters,
    triangulate_correspondences,
    build_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    Correspondences
)
from _corners import FrameCorners

triang_params = TriangulationParameters(
    max_reprojection_error=7.5,
    min_triangulation_angle_deg=1.0,
    min_depth=0.1)


def update_best_ans(max_points_count, best_known_views, corrs, ess_mat, intrinsic_mat, i, j):
    view_mat1 = np.eye(3, 4)
    r1, r2, t = cv2.decomposeEssentialMat(ess_mat)
    for view_mat2 in [np.hstack((r, t1)) for r in [r1, r2] for t1 in [t, -t]]:
        if max_points_count > 2000:
            return max_points_count, best_known_views, True
        points, _, _ = triangulate_correspondences(corrs, view_mat1, view_mat2,
                                                   intrinsic_mat, triang_params)
        if len(points) > max_points_count:
            best_known_views = ((i, view_mat3x4_to_pose(
                view_mat1)), (j, view_mat3x4_to_pose(view_mat2)))
            max_points_count = len(points)
    return max_points_count, best_known_views, False


def homography_validate(corrs, homography_threshold=0.65):
    hom_mat, mask = cv2.findHomography(corrs.points_1,
                                       corrs.points_2,
                                       method=cv2.RANSAC)
    return np.count_nonzero(mask) / len(corrs.ids) > homography_threshold


def find_known_views(intrinsic_mat, corner_storage):
    max_points_count, best_known_views = -1, ((None, None), (None, None))
    for i in range(len(corner_storage)):
        for j in range(i + 1, len(corner_storage)):
            corrs = build_correspondences(
                corner_storage[i], corner_storage[j])
            if len(corrs[0]) < 200:
                break
            ess_mat, mask = cv2.findEssentialMat(corrs.points_1, corrs.points_2,
                                                 cameraMatrix=intrinsic_mat)
            mask = (mask.squeeze() == 1)
            corrs = Correspondences(
                corrs.ids[mask], corrs.points_1[mask], corrs.points_2[mask])
            if ess_mat is None or homography_validate(corrs):
                continue

            max_points_count, best_known_views, flag = update_best_ans(
                max_points_count, best_known_views, corrs, ess_mat, intrinsic_mat, i, j)
            if flag:
                return best_known_views
    return best_known_views


def triangulate_and_add_points(corners1, corners2, view_mat1, view_mat2, intrinsic_mat, point_cloud_builder):
    corrs = build_correspondences(corners1, corners2)
    if len(corrs.points_1) == 0:
        return
    points3d, ids, _ = triangulate_correspondences(
        corrs,
        view_mat1, view_mat2,
        intrinsic_mat, triang_params)
    point_cloud_builder.add_points(ids, points3d)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    random.seed(1337)

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = find_known_views(
            intrinsic_mat, corner_storage)

    correspondences = build_correspondences(
        corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    view_mat_1 = pose_to_view_mat3x4(known_view_1[1])
    view_mat_2 = pose_to_view_mat3x4(known_view_2[1])
    points, ids, _ = triangulate_correspondences(correspondences,
                                                 view_mat_1, view_mat_2,
                                                 intrinsic_mat,
                                                 triang_params)

    view_mats = [None] * len(rgb_sequence)
    view_mats[known_view_1[0]] = view_mat_1
    view_mats[known_view_2[0]] = view_mat_2

    point_cloud_builder = PointCloudBuilder(ids, points)
    was_update = True
    while was_update:
        was_update = False
        for i, (_, corners) in enumerate(zip(rgb_sequence, corner_storage)):
            if view_mats[i] is not None:
                continue
            _, (idx_1, idx_2) = snp.intersect(point_cloud_builder.ids.flatten(),
                                              corners.ids.flatten(),
                                              indices=True)
            try:
                _, rvec, tvec, inliers = cv2.solvePnPRansac(point_cloud_builder.points[idx_1],
                                                            corners.points[idx_2],
                                                            intrinsic_mat,
                                                            distCoeffs=None)
                inliers = np.array(inliers, dtype=int)
                if len(inliers) > 0:
                    view_mats[i] = rodrigues_and_translation_to_view_mat3x4(
                        rvec, tvec)
                    was_update = True
                print(
                    f"\rProcessing {i} frame of {len(rgb_sequence)}. Number of inliers: {len(inliers)}", end="")
            except Exception:
                print(
                    f"\rProcessing {i} frame of {len(rgb_sequence)}. Number of inliers: {0}", end="")
            if view_mats[i] is None:
                continue
            cur_corner = filter_frame_corners(
                corner_storage[i], inliers.flatten())

            for j in range(len(rgb_sequence)):
                if view_mats[j] is None:
                    continue
                correspondences = build_correspondences(
                    corner_storage[j], cur_corner)
                if len(correspondences.ids) == 0:
                    continue
                points, ids, _ = triangulate_correspondences(correspondences,
                                                             view_mats[j], view_mats[i],
                                                             intrinsic_mat,
                                                             triang_params)
                point_cloud_builder.add_points(ids, points)

    for i in range(0, len(view_mats)):
        if view_mats[i] is None:
            print('Not all frames are processed. Death...')
            exit(1)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
