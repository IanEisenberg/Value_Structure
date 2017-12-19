#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 08:56:52 2017

@author: ian
"""

"""
generic task using psychopy
"""
from BaseExp import BaseExp
import json
import numpy as np
from psychopy import visual, core, event
from random import sample

class RLTask(BaseExp):
    """ class defining a probabilistic context task
    """
    
    def __init__(self,subjid, save_dir, stim_files, graph, 
                 trials, fullscreen = False):
        super(RLTask, self).__init__(subjid, save_dir, fullscreen)
        # set up "holder" variables
        self.RLdata = []  
        self.trialnum = 0
        self.pointtracker = 0
        self.correct_tracker = 0 # used to determine when to switch stim set
        self.correct_thresh = 4
        self.startTime = []
        
        # set up argument variables
        self.graph = graph
        self.stim_files = stim_files
        self.all_trials = trials
        
        # set up static variables
        self.action_keys = ['left','right']
            
    #**************************************************************************
    # ******* Display Functions **************
    #**************************************************************************
    
    def get_labeled_banner(self, labeled_stims, positions, 
                           height, display_value=True):
        banner = []
        labeled_stims = sample(labeled_stims, len(labeled_stims))
        for i, labeled_stim in enumerate(labeled_stims):
            stim_file, value = labeled_stim
            # stimulus
            stim = visual.ImageStim(self.win, image=stim_file,
                                units='norm', 
                                pos=(positions[i],height),
                                size=self.stim_size*.6)
            banner.append(stim)
            if display_value:
                # value
                valuestim = visual.TextStim(self.win, '%s RMB' % value, 
                                           pos=(positions[i],height-.3), 
                                           units='norm')
                banner.append(valuestim)
        return banner

    def place_labeled_stims(self, labeled_stims, scale):        
        # figure out position of shapes
        pos_limits = np.array([-.3,.3])*scale.stretch
        limits = [scale.low, scale.high]
        banner = []
        positions=[]
        for i, labeled_stim in enumerate(labeled_stims):
            overlap=0
            stim_file, value = labeled_stim
            # stimulus
            stim_pos = float(value-limits[0])/(limits[1]-limits[0])
            stim_pos = stim_pos*pos_limits[1]+(1-stim_pos)*pos_limits[0]
            if np.any([abs(i-stim_pos)<.1 for i in positions]):
                overlap = .18
            stim = visual.ImageStim(self.win, image=stim_file,
                                units='norm', 
                                pos=(stim_pos,
                                     scale.pos[1]+.17+overlap),
                                size=self.stim_size*.4)
            banner.append(stim)
            positions.append(stim_pos)
        return banner
        
    def presentStim(self, stim_files, textstim=None, duration=None):
        size = self.stim_size
        # present stim
        positions = [(-.3,0), (.3,0)]
        stim1 = visual.ImageStim(self.win, 
                                image=stim_files[0],
                                units='norm',
                                pos=positions[0],
                                size = size)
        stim2 = visual.ImageStim(self.win, 
                                image=stim_files[1],
                                units='norm',
                                pos=positions[1],
                                size = size)
        def draw_stim(other_stim=None):
            stim1.draw(); stim2.draw()
            if other_stim:
                other_stim.draw()
            if textstim:
                textstim.draw()
            self.win.flip()
        draw_stim()
        # get response
        recorded_keys = []
        stim_clock = core.Clock()
        if duration:
            while stim_clock.getTime() < duration:
                keys = event.getKeys(self.action_keys + [self.quit_key],
                                     timeStamped=stim_clock)
                for key,response_time in keys:
                    self.checkRespForQuitKey(key)
                    if len(recorded_keys) == 0:
                        choice = self.action_keys.index(key)
                        choice_box = visual.Circle(self.win,
                                                 size=size,
                                                 lineWidth=10,
                                                 pos=positions[choice])
                        draw_stim(other_stim = choice_box)
                    recorded_keys+=keys
        else:
            while len(recorded_keys) == 0:
                keys = event.getKeys(self.action_keys + [self.quit_key],
                                     timeStamped=stim_clock)
                for key,response_time in keys:
                    self.checkRespForQuitKey(key)
                    recorded_keys+=keys
                    
        return recorded_keys
        
    def presentTrial(self, trial):
        """
        This function presents a stimuli, waits for a response, tracks the
        response and RT and presents appropriate feedback. This function also controls the timing of FB 
        presentation.
        """
        trialClock = core.Clock()
        self.trialnum += 1
        trial['rewarded'] = np.nan
        trial['onset']=core.getTime() - self.startTime
        trial['response'] = np.nan
        trial['rt'] = np.nan
        trial['secondary_responses'] = []
        trial['secondary_rts'] = []
        trial['trialnum'] = self.trialnum
        # present stimulus and get response
        event.clearEvents()
        trialClock.reset()
        keys = self.presentStim(trial['stim_files'], 
                                duration = trial['duration'])
        if len(keys)>0:
            first_key = keys[0]
            # which did they choose
            choice = self.action_keys.index(first_key[0])
            # record response
            trial['response'] = choice
            trial['rt'] = first_key[1]
            # record any responses after the first
            trial['secondary_responses']=[i[0] for i in keys[1:]]
            trial['secondary_rts']=[i[1] for i in keys[1:]]
            trial['rewarded'] = trial['rewards'][choice]
            trial['correct'] = trial['correct_choice'] == choice
            self.pointtracker += trial['rewarded']
            if trial['correct']:
                self.correct_tracker += 1
            else:
                self.correct_tracker = 0
            if trial['rewarded']:
                self.presentTextToWindow('+1 point', color=[0,1,0])
            else:
                self.presentTextToWindow('+0 points', color=[1,0,0])
            print('Correct?: %s' % trial['correct'])
        else:
            self.presentTextToWindow('Missed Response', color=[1,1,1])
        core.wait(trial['feedback_duration'])
        self.clearWindow()    
        # log trial and add to data
        self.writeToLog(json.dumps(trial))
        self.RLdata.append(trial)
        return trial
    
 

                            
    def run_RLtask(self):
        # start graph learning
        self.presentTextToWindow('Get Ready!', duration=2)
        self.clearWindow()
        
        #
        for i, trial_set in enumerate(self.all_trials[:-1]):
            if i == 3:
                self.presentInstruction(
                        """
                        Take a break!
                                        
                        Press 5 when you are ready to continue
                        """)
            for i,trial in enumerate(trial_set):
                self.presentTrial(trial)
                if self.correct_tracker >= self.correct_thresh:
                    break
        # another break
        self.presentInstruction(
                    """
                    Take a break!
                                    
                    Press 5 when you are ready to continue
                    """)
        # final trials
        for i,trial in enumerate(trial_set[-1]):
                self.presentTrial(trial)
        
    def run_task(self, pause_trials = None):
        self.setupWindow()
        self.stim_size = self.getSquareSize(self.win)        
        # instructions
        
        self.presentInstruction(
            """
            In this task, you will be shown two stimuli at a time
            from the last task. 
    
            
            Each stimuli has a chance of earning 1 point when you select it.
            The chance of earning a point is different for each stimulus:
            Some stimuli have a higher chance to earn a point than others.
            
            Your goal in this task is to get as many points as possible. Each
            trial is short, so please respond as quickly while trying to
            pick the more rewarding shape.
            
            Press 5 to continue...
            """)
                
        self.run_RLtask()
        
        # clean up and save
        taskdata = {
                'graph': self.graph,
                'action_keys': self.action_keys
                } 
        otherdata = {'RLdata': self.RLdata,
                     'total_points': self.pointtracker}
        self.writeData(taskdata, otherdata)
        return self.pointtracker

