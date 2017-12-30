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
from psychopy import prefs
prefs.general['audioLib'] = ['sounddevice']
from psychopy import visual, core, event, sound
from random import sample

# play a sound at beginning to ensure that psychopy's sound is working
error_sound = sound.Sound(secs=.1,value=500)
miss_sound = sound.Sound(secs=.1,value=700)

class RotationStructureTask(BaseExp):
    """ class defining a probabilistic context task
    """
    
    def __init__(self, expid, subjid, save_dir, stim_files, graph, 
                 trials, familiarization_trials, 
                 fullscreen = False):
        # set up "holder" variables
        self.structuredata = []  
        self.pointtracker = 0
        self.trialnum = 0
        self.startTime = []
        
        # set up argument variables
        self.familiarization_trials = familiarization_trials
        self.graph = graph
        self.stim_files = stim_files
        self.trials = trials
        
        # set up static variables
        self.action_keys = ['z','m']
        np.random.shuffle(self.action_keys)
        self.n_value_ratings = 3
        self.test_familiarization = True
        # init Base Exp
        super(RotationStructureTask, self).__init__(expid, subjid, save_dir, fullscreen)
            
    #**************************************************************************
    # ******* Display Functions **************
    #**************************************************************************

    def presentStim(self, stim_file, rotation, allowed_keys=None, textstim=None, 
                    duration=None, correct_choice=None):
        if allowed_keys is None:
            allowed_keys = self.action_keys
        size = self.stim_size
        # present stim
        if rotation==90:
            size = size[::-1]
        stim = visual.ImageStim(self.win, 
                                image=stim_file,
                                ori=rotation,
                                units='norm',
                                size = size)
        stim.draw()
        if textstim:
            textstim.draw()
        self.win.flip()
        # get response
        recorded_keys = []
        stim_clock = core.Clock()
        if duration:
            while stim_clock.getTime() < duration:
                keys = event.getKeys(allowed_keys + [self.quit_key],
                                     timeStamped=stim_clock)
                for key,response_time in keys:
                    self.checkRespForQuitKey(key)
                    if correct_choice:
                        if key!=correct_choice and len(recorded_keys)==0:
                            error_sound.play()
                    recorded_keys+=keys
        else:
            while len(recorded_keys) == 0:
                keys = event.getKeys(allowed_keys+ [self.quit_key],
                                     timeStamped=stim_clock)
                for key,response_time in keys:
                    self.checkRespForQuitKey(key)
                    if correct_choice:
                        if key!=correct_choice and len(recorded_keys)==0:
                            error_sound.play()
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
        trial['onset']=core.getTime() - self.startTime
        trial['response'] = np.nan
        trial['rt'] = np.nan
        trial['secondary_responses'] = []
        trial['secondary_rts'] = []
        trial['trialnum'] = self.trialnum
        correct_choice = self.action_keys[[0,90].index(trial['rotation'])]
        # present stimulus and get response
        event.clearEvents()
        trialClock.reset()
        keys = self.presentStim(trial['stim_file'], 
                                trial['rotation'],
                                duration = trial['duration'],
                                correct_choice=correct_choice)
        if len(keys)>0:
            first_key = keys[0]
            choice = ['unrot','rot'][self.action_keys.index(first_key[0])]
            # record response
            trial['response'] = choice
            trial['rt'] = first_key[1]
            # record any responses after the first
            trial['secondary_responses']=[i[0] for i in keys[1:]]
            trial['secondary_rts']=[i[1] for i in keys[1:]]
            if correct_choice == first_key[0]:
                trial['correct']=True
                # record points for bonus
                self.pointtracker += 1
            else:
                trial['correct']=False
        else:
            miss_sound.play()
            core.wait(.5)
                
        # log trial and add to data
        self.writeToLog(json.dumps(trial))
        self.structuredata.append(trial)
        return trial
    
    
    def run_familiarization(self):
        i=0
        stims = sample(self.stim_files, len(self.stim_files))
        while i<len(stims):
            filey = stims[i]
            keys = self.presentStim(filey, 0, ['left','right'])
            if keys[0][0] == 'right':
                i+=1
            elif i>0:
                i-=1
                
    def run_familiarization_test(self):
        np.random.shuffle(self.familiarization_trials)
        # ensure some rotation
        N = len(self.familiarization_trials)
        rotation = np.zeros(N)
        rotation[:int(N*.2)] = 90
        np.random.shuffle(rotation)
        for i, trial in enumerate(self.familiarization_trials):
            trial['rotation'] = rotation[i]
            self.presentTrial(trial)
                            
    def run_graph_learning(self):
        self.trialnum = 0
        # start graph learning
        pause_trials = (len(self.trials)/3, len(self.trials)/3*2)
        self.presentTextToWindow('Get Ready!', duration=2)
        self.clearWindow()
        for trial in self.trials:
            self.presentTrial(trial)
            if self.trialnum in pause_trials:
                self.presentInstruction(
                        """
                        Take a break!
                                        
                        Press 5 when you are ready to continue
                        """)
        
    def run_task(self, pause_trials = None):
        self.setupWindow()
        self.stim_size = self.getSquareSize(self.win)
        self.startTime = core.getTime()
        self.presentInstruction(
            """
            Welcome! 
            
            This experiment has two parts. 
            
            Each part will last around 30 minutes.
            
            Press 5 to continue...
            """)
        
        # instructions
        intro_text = """
            In the first part of this study, abstract images
            will be shown one at a time for a short amount of time.
            
            Your task is to indicate whether 
            each stimulus is rotated or unrotated.
            
            %s key: Unrotated
            %s key: Rotated
            
            You will hear a beep if you choose incorrectly 
            or miss a response.
            
            Press 5 to [blank]
            """
            
        self.presentInstruction(intro_text.replace('[blank]',
                                                   'hear the error beep') 
                                    % (self.action_keys[0].title(), 
                                       self.action_keys[1].title()))
        
        # show beeps
        error_sound.play()
        
        self.presentInstruction(intro_text.replace('[blank]', 
                                                   'hear the miss beep') 
                                    % (self.action_keys[0].title(), 
                                       self.action_keys[1].title()))
        miss_sound.play(); core.wait(.5)
        
        self.presentInstruction(
            """
            We will start by familiarizing you with the 
            images. Each of these images is unrotated.
            
            Press the left and right keys to move through the images.
            
            Press 5 to continue...
            """)
        self.run_familiarization()
        
        
        if self.test_familiarization == True:
            learned=False
            num_misses = 0
            self.presentInstruction(
                """
                We will now practice responding to the images. 
                Indicate whether the stimulus is unrotated or rotated.
                
                        %s key: Unrotated
                        %s key: Rotated
                        
                Wait for the experimenter.
                """ % (self.action_keys[0].title(), 
                       self.action_keys[1].title()))
            while not learned:     
                self.run_familiarization_test()
                acc = np.mean([t['correct'] for t in self.structuredata 
                               if t['exp_stage'] == 'familiarization_test'])
                if acc>.8 or num_misses>6:
                    learned=True
                else:
                    num_misses += 1
                    if num_misses==4:
                        self.presentInstruction(
                            """
                            Seems you could use a refresher! Please look over the
                            images again and try to remember which way the stimulus
                            is unrotated
                            
                            Press left and right keys to move through the images
                            
                            Press 5 to continue...
                            """)
                        self.run_familiarization()
                        self.presentInstruction(
                            """
                            We will now practice responding to the images again.
                            
                                %s key: Unrotated
                                %s key: Rotated
                                
                            Press 5 to continue...
                            """ % (self.action_keys[0].title(), 
                                   self.action_keys[1].title()))
                
        # structure learning 
        self.presentInstruction(
            """
            Finished with familiarization. In the next section, 
            indicate whether the image is unrotated or rotated.
            
                %s key: Unrotated
                %s key: Rotated
            
            Each image will only come up on the screen for a short 
            amount of time. Please respond as quickly and accurately 
            as possible.
            
            There will be two breaks.
            
            Wait for the experimenter
            """ % (self.action_keys[0].title(), self.action_keys[1].title()))
        
        self.run_graph_learning()
        
        # clean up and save
        taskdata = {
                'stim_files': self.stim_files,
                'graph': self.graph,
                'action_keys': self.action_keys
                } 
        otherdata = {'structuredata': self.structuredata}
        self.writeData(taskdata, otherdata)
        
        self.presentInstruction(
            """
            Done with the first task! Take a break for a few minutes.
            
            Please wait for the experimenter.
            """)

