# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 
import math
from scipy.stats import chi2

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        
        # the following only works for at most one track and one measurement
        association_matrix = []
        self.unassigned_tracks = [] # reset lists
        self.unassigned_meas = []
#         
#         if len(meas_list) > 0:
#             self.unassigned_meas = [0]
#         if len(track_list) > 0:
#             self.unassigned_tracks = [0]
#         if len(meas_list) > 0 and len(track_list) > 0: 
#             self.association_matrix = np.matrix([[0]])
        
        for track in track_list:
            res = []
            for meas in meas_list:
                MHD = self.MHD(track, meas, KF)
                sensor = meas.sensor
                if self.gating_ok(MHD, sensor):
                    res.append(MHD)
#                     if sensor.name == "camera":
#                         print("track {}, state={}, MHD= {}".format(track.id, track.state, MHD))
                else:
                    res.append(np.inf)
            association_matrix.append(res)
        #Here we set all tracks and measurements to be unassigned, later on 
        #we will finally assign them when calling get_closest_track_and_meas
        self.unassigned_tracks = np.arange(len(track_list)).tolist()
        self.unassigned_meas = np.arange(len(meas_list)).tolist()
        
        self.association_matrix = np.matrix(association_matrix)
        
        return
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # the following only works for at most one track and one measurement  
        A = self.association_matrix
        if np.min(A) == np.inf:
            return np.nan, np.nan

        # get indices of minimum entry
        ij_min = np.unravel_index(np.argmin(A, axis=None), A.shape) 
        ind_track = ij_min[0]
        ind_meas = ij_min[1]

        # delete row and column for next update
        A = np.delete(A, ind_track, 0) 
        A = np.delete(A, ind_meas, 1)
        self.association_matrix = A

        # update this track with this measurement
        update_track = self.unassigned_tracks[ind_track] 
        update_meas = self.unassigned_meas[ind_meas]

        # remove this track and measurement from list
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)

        return update_track, update_meas     

    def gating_ok(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        df = None
        gate_val = None
        if sensor.name == 'lidar':
            #While fine tuning the algorihm, we find that it's better to have a larger gate threshold for lidar 
            #which means current lidar noise is a bit underestimated
            df = 2 
            gate_val = params.gating_threshold_lidar
        
        if sensor.name == 'camera':
            gate_val = params.gating_threshold
            df = 1
        x= MHD * MHD
        per = chi2.cdf(x, df)
        if sensor.name == 'lidar':
            print("lidar chisqr = {}".format(per))
        if per <  gate_val:
            return True
        else:
            return False
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        z = np.matrix(meas.z)
        z_pred = meas.sensor.get_hx(track.x)
        y = z - z_pred 
        S = meas.R
        
        d = math.sqrt(y.T * S.I * y)
        
        
        return d
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        if len(meas_list) == 0:
            return
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            if meas_list[0].sensor.name == 'lidar':
                manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        if meas_list[0].sensor.name == 'lidar':
            manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score, 'state={}'.format(track.state))