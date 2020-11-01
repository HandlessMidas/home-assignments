#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    eye3x4,
    to_camera_center,
    calc_inlier_indices,
    check_baseline
)


def solve_pnp(frame, intrinsic_mat, corner_storage, point_cloud_builder, outliers, view_mats, time=0):
    corners = corner_storage[frame]
    inter, object_id, image_id = np.intersect1d(
        point_cloud_builder.ids, corners.ids, return_indices=True)

    for mask in [outliers, np.zeros_like(outliers)]:
        ids = np.where(np.invert(mask))
        inliers_ids = np.intersect1d(ids, inter, return_indices=True)[2]
        image_ids = corners.ids[image_id[inliers_ids]]
        image_points = corners.points[image_id[inliers_ids]]
        object_points = point_cloud_builder.points[object_id[inliers_ids]]
        if object_points.shape[0] >= 4:
            break

    _, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points, imagePoints=image_points, cameraMatrix=intrinsic_mat, distCoeffs=None, reprojectionError=2, flags=cv2.SOLVEPNP_EPNP)
    if time == 0:
        triang_error = 4
        while len(inliers) < 5 and triang_error < 100:
            triang_error *= 1.03
            inliers = calc_inlier_indices(object_points, image_points, np.matmul(
                intrinsic_mat, view_mats[frame]), triang_error)
        outlier_ids = np.setdiff1d(image_ids, image_ids[inliers])
        outliers[outlier_ids] = True
        solve_pnp(frame, intrinsic_mat, corner_storage,
                  point_cloud_builder, outliers, view_mats, time=1)
    print(f'Number of inliers: {len(inliers)}')
    return rodrigues_and_translation_to_view_mat3x4(rvec, tvec)


def add_points(frame_1, frame_2, corner_storage, view_mats, point_cloud_builder, intrinsic_mat):
    triang_params = TriangulationParameters(4, 1, 0)
    cors = build_correspondences(
        corner_storage[frame_1], corner_storage[frame_2], ids_to_remove=point_cloud_builder.ids)
    if (len(cors.ids) == 0):
        return
    points, ids, _ = triangulate_correspondences(
        cors, view_mats[frame_1], view_mats[frame_2], intrinsic_mat, triang_params)
    point_cloud_builder.add_points(ids, points)
    print(f'Point cloud size: {point_cloud_builder.points.size}')


def track_camera(corner_storage, intrinsic_mat, known_view_1, known_view_2):
    point_cloud_builder = PointCloudBuilder()
    total_frames = len(corner_storage)
    view_mats = [eye3x4()] * total_frames
    view_mat_1 = pose_to_view_mat3x4(known_view_1[1])
    view_mat_2 = pose_to_view_mat3x4(known_view_2[1])
    view_mats[known_view_1[0]] = view_mat_1
    view_mats[known_view_2[0]] = view_mat_2
    known_frame_1 = min(known_view_1[0], known_view_2[0])
    known_frame_2 = max(known_view_1[0], known_view_2[0])
    processing_frame = known_frame_1 + 1
    outliers = np.zeros(
        max([i.ids.size for i in corner_storage]) + 1, dtype=bool)
    camera_centers = [to_camera_center(
        view_mat_1), to_camera_center(view_mat_2)]
    start_distance = np.linalg.norm(camera_centers[0] - camera_centers[1])
    last_frame = known_view_1[0]

    add_points(known_frame_1, known_frame_2, corner_storage,
               view_mats, point_cloud_builder, intrinsic_mat)

    for _ in range(2, total_frames):
        processing_frame += ((processing_frame ==
                              known_frame_2) and (known_frame_2 != total_frames - 1))

        print(f'Processing {processing_frame}/{total_frames - 1} frame')

        if processing_frame < total_frames:
            view_mats[processing_frame] = solve_pnp(
                processing_frame, intrinsic_mat, corner_storage, point_cloud_builder, outliers, view_mats)

            counter = 0
            last_frame_dist = 3
            while counter < 3:
                if processing_frame > known_frame_1:
                    last_frame = processing_frame - last_frame_dist
                else:
                    last_frame = processing_frame + last_frame_dist
                last_frame_dist += 1
                if last_frame < total_frames and (known_frame_1 <= last_frame or processing_frame < known_frame_1):
                    if check_baseline(view_mats[last_frame], view_mats[processing_frame], start_distance * 0.1):
                        add_points(last_frame, processing_frame, corner_storage,
                                   view_mats, point_cloud_builder, intrinsic_mat)
                        counter += 1
                else:
                    break

        if processing_frame > known_frame_1:
            processing_frame += 1
        else:
            processing_frame -= 1

    return view_mats, point_cloud_builder


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    view_mats, point_cloud_builder = track_camera(
        corner_storage, intrinsic_mat, known_view_1, known_view_2)

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
