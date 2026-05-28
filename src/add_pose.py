import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # The robot rotates ~45 degrees anti-clockwise, moves ~2 meters forward,
    # then rotates ~45 degrees more anti-clockwise.
    # In X(3)'s local frame: dx = 2*cos(45°) = sqrt(2), dy = 2*sin(45°) = sqrt(2), dtheta = pi/2
    odometry = gtsam.Pose2(math.sqrt(2), math.sqrt(2), math.pi / 2)

    # Add odometry factor between X(3) and X(4)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    # Reconstruct the nominal X(3) by composing the prior and previous odometry steps:
    # X(1)=(0,0,0) -> +2m -> X(2)=(2,0,0) -> +2m -> X(3)=(4,0,0)
    nominal_x3 = gtsam.Pose2(0.0, 0.0, 0.0)
    nominal_x3 = nominal_x3.compose(gtsam.Pose2(2.0, 0.0, 0.0))
    nominal_x3 = nominal_x3.compose(gtsam.Pose2(2.0, 0.0, 0.0))

    # Compose with our odometry to get the initial estimate for X(4)
    pose4 = nominal_x3.compose(odometry)
    initial_estimate.insert(X(4), pose4)

    return graph, initial_estimate