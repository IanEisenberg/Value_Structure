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

class StructureTask(BaseExp):
    """ class defining a probabilistic context task
    """
    
    def __init__(self,subjid, save_dir, stim_files, graph, 
                 trials, familiarization_trials, 
                 fullscreen = False):
        super(StructureTask, self).__init__(subjid, save_dir, fullscreen)
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
        self.action_keys = ['left','right']
        np.random.shuffle(self.action_keys)
        self.n_value_ratings = 3
        self.test_familiarization = False
            
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
        
    def presentStim(self, stim_file, rotation, textstim=None, duration=None,
                    correct_choice=None):
        error_sound = sound.Sound(secs=.1,value=500)
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
                keys = event.getKeys(self.action_keys + [self.quit_key],
                                     timeStamped=stim_clock)
                for key,response_time in keys:
                    self.checkRespForQuitKey(key)
                    if correct_choice:
                        if key!=correct_choice and len(recorded_keys)==0:
                            error_sound.play()
                    recorded_keys+=keys
        else:
            while len(recorded_keys) == 0:
                keys = event.getKeys(self.action_keys + [self.quit_key],
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
        trial['correct'] = False
        trial['onset']=core.getTime() - self.startTime
        trial['response'] = -1
        trial['rt'] = -1
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
            miss_sound = sound.Sound(secs=.1,value=700)
            miss_sound.play()
            core.wait(.5)
                
        # log trial and add to data
        self.writeToLog(json.dumps(trial))
        self.structuredata.append(trial)
        return trial
    
    
    def run_familiarization(self):
        i=0
        stims = sample(self.stim_files, len(self.stim_files))
        while i<len(stims)*2:
            filey = stims[i//2]
            text = ['Unrotated','Rotated'][i%2==1]
            textstim = visual.TextStim(self.win, text, pos=[0,.5], units='norm')
            keys = self.presentStim(filey, [0,90][i%2==1], textstim)
            if keys[0][0] == 'right':
                i+=1
            elif i>0:
                i-=1
                
    def run_familiarization_test(self):
        np.random.shuffle(self.familiarization_trials)
        for trial in self.familiarization_trials:
            trial['rotation'] = np.random.choice([0,90])
            self.presentTrial(trial)
                            
    def run_graph_learning(self):
        # show beeps
        self.presentInstruction('Press 5 to hear the error beep')
        error_sound = sound.Sound(secs=.1,value=500)
        error_sound.play(); core.wait(.5)
        self.presentInstruction('Press 5 to hear the "miss" beep')
        error_sound = sound.Sound(secs=.1,value=700)
        error_sound.play(); core.wait(.5)
        self.presentInstruction('Press 5 to start')
        # start graph learning
        pause_trials = (len(self.trials)/3, len(self.trials)/3*2)
        self.presentTextToWindow('Get Ready!', duration=2)
        self.clearWindow()
        for i,trial in enumerate(self.trials):
            self.presentTrial(trial)
            if i in pause_trials:
                self.presentInstruction(
                        """
                        Take a break!
                                        
                        Press 5 when you are ready to continue
                        """)
        
    def run_task(self, pause_trials = None):
        self.setupWindow()
        self.stim_size = self.getSquareSize(self.win)
        self.presentInstruction('Welcome! Press 5 to continue...')
        
        # instructions
        
        self.presentInstruction(
            """
            This task will help us learn about how you value things.
            
            There are two parts of this task. In the first part
            you will interact with 11 different stimuli.
            
            In the second phase you will provide a
            value for the stimuli.
            
            Press 5 to continue...
            """)
                
        self.presentInstruction(
            """
            In the first part of this study, stimuli
            will be shown one at a time for a short 
            amount of time.
            
            Your task is to indicated whether 
            each stimulus is rotated or unrotated.
            
            We will start by familiarizing you with the 
            stimuli. Press the left and right keys to 
            move through the stimuli.
            
            Press 5 to continue...
            """)
        
        if self.test_familiarization == True:
            learned=False
            while not learned:
                self.run_familiarization()
                self.presentInstruction(
                    """
                    We will now practice responding to the stimuli. 
                    Indicate whether the stimulus is unrotated or rotated.
                    
                            %s key: Unrotated
                            %s key: Rotated
                            
                    Press 5 to continue...
                    """ % (self.action_keys[0], self.action_keys[1]))
                self.run_familiarization_test()
                acc = np.mean([t['correct'] for t in self.structuredata 
                               if t['exp_stage'] == 'familiarization_test'])
                if acc>.75:
                    learned=True
                else:
                    self.presentInstruction(
                        """
                        Seems you could use a refresher! Please look over the
                        stimuli again and try to remember which way the stimulus
                        is unrotated
                        
                        Press 5 to continue...
                        """)
        else:
            self.run_familiarization()
                
        # structure learning 
        self.presentInstruction(
            """
            Finished with familiarization. In the next section, 
            indicated whether the stimulus is unrotated or rotated.
            
                %s key: Unrotated
                %s key: Rotated
            
            Each stimulus will only come up on the screen for a short 
            amount of time. Please respond as quickly and accurately 
            as possible.
            
            You will hear a beep if you choose incorrectly or miss
            a response.
            
            This section takes a long time, so there will be two
            breaks.
            
            Press 5 to continue...
            """ % (self.action_keys[0], self.action_keys[1]))
        
        self.run_graph_learning()
        
        # clean up and save
        taskdata = {
                'graph': self.graph,
                'action_keys': self.action_keys
                } 
        otherdata = {'structuredata': self.structuredata}
        self.writeData(taskdata, otherdata)

