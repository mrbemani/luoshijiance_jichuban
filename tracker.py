# -*- coding: utf-8 -*-
# Tracker Using Hungarian Algorithm


__author__ = 'Mr.Bemani'

# Import python libraries
import numpy as np


# predict the next possible position based on past positions
def predict_next_position(past_positions):
    # Convert past positions to a numpy array
    past_positions = np.array(past_positions)
    
    # Calculate the average displacement between consecutive positions
    displacements = np.diff(past_positions, axis=0)
    average_displacement = np.mean(displacements, axis=0)
    
    # Predict the next position by adding the average displacement to the last known position
    last_position = past_positions[-1]
    next_position = last_position + average_displacement
    
    return next_position


# use cupy if possible
try:
    from cupy.optimize import linear_sum_assignment
except:
    print ("no cupy, use scipy")
    from scipy.optimize import linear_sum_assignment
from datetime import datetime

class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path
        self.start_time = datetime.utcnow()
        self.passed = False


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def update(self, detections, allow_moving_up:bool=True, move_up_thresh:int=0):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # Create tracks if no tracks vector found
        if len(self.tracks) == 0:
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = self.tracks[i].prediction - detections[j]
                    distance = np.sqrt(diff[0]*diff[0] +
                                       diff[1]*diff[1])
                    
                    cost[i][j] = distance
                    # does not allow moving up
                    if distance > self.dist_thresh:
                        cost[i][j] = 1e4
                    if not allow_moving_up and diff[1] > move_up_thresh:
                        cost[i][j] = 1e4
                    # check if the trace is moving up
                    if len(self.tracks[i].trace) >= 3 and not allow_moving_up:
                        if self.tracks[i].trace[-1][1][0] < detections[-3][1][0] - move_up_thresh:
                            cost[i][j] = 1e4
                except:
                    pass

        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = [-1 for _ in range(N)]
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if assignment[i] != -1:
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if cost[i][assignment[i]] > self.dist_thresh:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = [i for i in range(len(self.tracks)) if self.tracks[i].skipped_frames > self.max_frames_to_skip]
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
        
        # Now look for un_assigned detects
        un_assigned_detects = [i for i in range(len(detections)) if i not in assignment]
        
        # Start new tracks
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],
                              self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            if assignment[i] != -1:
                self.tracks[i].skipped_frames = 0
                if len(self.tracks[i].trace) > 1:
                    # predict the next position by giving the last 2 positions
                    # next_pos = predict_next_position(self.tracks[i].trace[-2:])
                    # assign to prediction
                    self.tracks[i].prediction = detections[assignment[i]]
                else:
                    self.tracks[i].prediction = detections[assignment[i]]

            if len(self.tracks[i].trace) > self.max_trace_length:
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
