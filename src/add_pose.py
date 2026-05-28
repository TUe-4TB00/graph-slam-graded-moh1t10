import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # The robot rotates ~45 degrees anti-clockwise, moves ~2 meters, then rotates ~45 degrees more.
    # Total heading change: 90 degrees (pi/2 radians)
    # In X(3)'s local frame, the 2m movement is at 45 degrees:
    #   dx = 2 * cos(45°) = sqrt(2), dy = 2 * sin(45°) = sqrt(2)
    dx = 2.0 * math.cos(math.pi / 4)   # ~1.4142
    dy = 2.0 * math.sin(math.pi / 4)   # ~1.4142
    dtheta = math.pi / 2                # 90 degrees total rotation

    # Add odometry factor between X(3) and X(4)
    graph.add(gtsam.BetweenFactorPose2(
        X(3), X(4),
        gtsam.Pose2(dx, dy, dtheta),
        ODOMETRY_NOISE
    ))

    # Optimize the existing graph to get a clean X(3) before composing
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate).optimize()
    pose3 = result.atPose2(X(3))
    pose4 = pose3.compose(gtsam.Pose2(dx, dy, dtheta))
    initial_estimate.insert(X(4), pose4)

    return graph, initial_estimate