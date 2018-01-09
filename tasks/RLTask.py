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
from utils.utils import (gen_random_RL_trials, gen_structured_RL_trials)

class RLTask(BaseExp):
    """ class defining a probabilistic context task
    """
    
    def __init__(self, expid, subjid, save_dir, stim_files, values, 
                 sequence_type = 'structured', fullscreen = False,
                 trial_params = {}):
        # set up "holder" variables
        self.RLdata = []  
        self.trialnum = 0
        self.pointtracker = 0
        self.startTime = []
        
        # set up argument variables
        self.values = values
        self.stim_files = stim_files
        self.sequence_type = sequence_type
        self.correct_tracker = 0 # used to determine when to switch stim set
        
        assert self.sequence_type in ['structured', 'random', 'semistructured']
        if self.sequence_type == 'structured':
            self.trials = gen_structured_RL_trials(stim_files, 
                                                       values, 
                                                       **trial_params)
        elif self.sequence_type == 'random':
            self.trials = gen_random_RL_trials(stim_files, 
                                                   values, 
                                                   **trial_params)           
        # set up static variables
        self.action_keys = ['left','right']
        # init Base Exp
        super(RLTask, self).__init__(expid, subjid, save_dir, fullscreen)
            
    #**************************************************************************
    # ******* Display Functions **************
    #**************************************************************************
    
    def draw_stims(self, stim_files, other_stim=None, textstim=None):
        assert len(stim_files)==2
        size = self.stim_size
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
        stim1.draw(); stim2.draw()
        if other_stim:
            other_stim.draw()
        if textstim:
            textstim.draw()
        self.win.flip()
        
    def presentStim(self, stim_files, textstim=None, duration=None):
        # present stim
        positions = [(-.3,0), (.3,0)]
        self.draw_stims(stim_files, textstim)
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
                                                 size=self.stim_size,
                                                 lineWidth=10,
                                                 pos=positions[choice],
                                                 edges=96)
                        self.draw_stims(stim_files, 
                                       other_stim=choice_box)
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
        trial['correct'] = np.nan
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
            if np.isnan(trial['correct_choice']) == False:
                trial['correct'] = trial['correct_choice'] == choice
            self.pointtracker += trial['rewarded']
            if trial['correct']:
                self.correct_tracker += 1
            else:
                self.correct_tracker = 0
            if trial['display_reward']:
                if trial['rewarded']:
                    self.presentTextToWindow('+1 point', color=[0,1,0])
                else:
                    self.presentTextToWindow('+0 points', color=[1,0,0])
            else:
                self.presentTextToWindow('Points Hidden!', color=[1,0,0])
        else:
            self.presentTextToWindow('Please Respond Faster', color=[1,1,1])
        core.wait(trial['feedback_duration'])
        self.clearWindow()    
        # log trial and add to data
        self.writeToLog(json.dumps(trial))
        self.RLdata.append(trial)
        return trial
                 
    def run_RLtask(self):
        timer_text = "Take a break!\n\nContinue in: \n\n       "
        self.startTime = core.getTime()
        # start graph learning
        self.presentTextToWindow('Get Ready!', duration=2)
        self.clearWindow()
        # use different scripts depending on whether the trials were generated
        # using generate_structured_RL_trials or generate_random_RL_trials
        pause_trials = (len(self.trials)/3, len(self.trials)/3*2)
        for trial in self.trials:
            self.presentTrial(trial)
            if self.trialnum in pause_trials:
                self.presentTimer(duration=30, text=timer_text)
                self.RLdata.append({'exp_stage': 'break',
                                    'duration': 30})
                self.presentInstruction("Press 5 to restart")
                self.presentTextToWindow('Get Ready!', duration=2)
            
    def run_task(self, pause_trials = None):
        self.setupWindow()
        self.stim_size = self.getSquareSize(self.win)        
        # instructions
        # introduction
        texts = []
        texts.append( \
            """
  In this task, two images from the last task will 
  be shown on the screen at once, as shown.
  
  You select one image by pressing the 
  correponding arrow key (left or right).\n\n\n\n\n\n\n\n\n\n\n
                  Press 5 to continue...
            """)
        
        texts.append( \
            """
  Each image has a chance of earning 1 point when you select it.
  The chance of earning a point is different for each image
  
  For example, one image may result in a point 80% of the time,
  while another image may result in a point 50% of the time.\n\n\n\n\n\n\n\n\n\n\n
                  Press 5 to continue...
            """)
            
            
        for text in texts:
            intro_stim=visual.TextStim(self.win, 
                                       text=text,
                                       font='BiauKai',
                                       height=.07,
                                       wrapWidth=100,
                                       pos=(0,.1))
            
            instruction_stims = [self.stim_files[0], self.stim_files[7]]
            self.draw_stims(instruction_stims, textstim=intro_stim)
            self.waitForKeypress(self.trigger_key, clear=False)

        self.presentInstruction(
            """
            Your goal in this task is to get as many points as possible.
            
            After you select an image, most of the time you will be told
            whether you won a point or not. Occasionally the points will
            be hidden. Even when they are hidden they count towards
            your total! 
            
            Each trial is short, so please respond quickly while trying 
            to pick the more rewarding shape.
            
            Wait for the experimenter
            """)
                
        self.run_RLtask()
        
        # clean up and save
        taskdata = {
                'stim_files': self.stim_files,
                'values': self.values,
                'action_keys': self.action_keys
                } 
        otherdata = {'RLdata': self.RLdata,
                     'total_points': self.pointtracker}
        self.writeData(taskdata, otherdata)
        
        self.presentInstruction(
            """
            Done with the second task! You earned %s points.
            
            Please wait for the experimenter.
            """ % str(self.pointtracker))
        self.closeWindow()
        return self.pointtracker

