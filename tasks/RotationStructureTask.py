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
from utils.utils import gen_rotstructure_trials

# play a sound at beginning to ensure that psychopy's sound is working
error_sound = sound.Sound(secs=.1,value=500)
miss_sound = sound.Sound(secs=.1,value=700)

class RotationStructureTask(BaseExp):
    """ class defining a probabilistic context task
    """
    
    def __init__(self, expid, subjid, save_dir, stim_files, graph, 
                 trial_params = {}, fullscreen = False):
        # set up default trial params
        self.trial_params = {'num_trials': 1400,
                             'proportion_rotated': .15,
                             'seed': None}
                             
        # set up "holder" variables
        self.structuredata = []  
        self.pointtracker = 0
        self.trialnum = 0
        self.startTime = []
        
        # set up argument variables
        self.graph = graph
        self.stim_files = stim_files
        self.trial_params.update(trial_params)
        self.trials = gen_rotstructure_trials(
                               self.graph, 
                               self.stim_files, 
                               self.trial_params['num_trials'], 
                               exp_stage='structure_learning',
                               proportion_rotated=self.trial_params['proportion_rotated'],
                               seed=self.trial_params['seed'])

        
        # set up static variables
        self.action_keys = ['z','m']
        np.random.shuffle(self.action_keys)
        # init Base Exp
        super(RotationStructureTask, self).__init__(expid, subjid, save_dir, fullscreen)
            
    #**************************************************************************
    # ******* Display Functions **************
    #**************************************************************************
    def draw_familiarization_stims(self, stim_file):
        rot_flip = np.random.randint(0,2)
        correct_key = ['left', 'right'][rot_flip]

        size = self.stim_size
        positions = [(-.3,0), (.3,0)]
        stim1 = visual.ImageStim(self.win, 
                                image=stim_file,
                                units='norm',
                                pos=positions[0],
                                size=size if not rot_flip else size[::-1],
                                ori=rot_flip*90)
        stim2 = visual.ImageStim(self.win, 
                                image=stim_file,
                                units='norm',
                                pos=positions[1],
                                size=size if rot_flip else size[::-1],
                                ori=(1-rot_flip)*90)
        stim1.draw(); stim2.draw()
        self.win.flip()
        return correct_key
        
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
        stims = []
        for i in range(4):
            tmp = self.stim_files[:]
            np.random.shuffle(tmp)
            stims.extend(tmp)
        while stims:
            stim = stims.pop(0)
            correct_key = self.draw_familiarization_stims(stim)
            keys = self.waitForKeypress(['left','right'])
            key = keys[0][0][0]
            print(key,correct_key, key==correct_key)
            if key != correct_key:
                print('error')
                stims.append(stim)
                error_sound.play()
            core.wait(.5)
                            
    def run_graph_learning(self):
        self.trialnum = 0
        # start graph learning
        pause_trials = [len(self.trials)//4*i for i in range(1,4)]
        timer_text = "Take a break!\n\nContinue in: \n\n       "
        self.presentTextToWindow('Get Ready!', duration=2)
        self.clearWindow()
        for trial in self.trials:
            self.presentTrial(trial)
            if self.trialnum in pause_trials:
                clock = core.Clock()
                self.presentTimer(duration=30, text=timer_text)
                self.presentInstruction("Press 5 to continue5")
                break_length = clock.getTime()
                self.structuredata.append({'exp_stage': 'break',
                                           'duration': break_length})
                self.presentTextToWindow('Get Ready!', duration=2)
        
    def run_task(self, pause_trials = None):
        self.setupWindow()
        self.stim_size = self.getSquareSize(self.win)
        self.startTime = core.getTime()
        self.presentInstruction(
            """
            Welcome! 
            
            This experiment has three parts. 
            
            This first part will take about 35 minutes.
            
            Press 5 to continue...
            """)
        
        # instructions
        intro_text = """
            In the first part of this study, abstract images
            will be shown one at a time for a short amount of time.
            
            Your task is to indicate whether 
            each image is rotated or unrotated.
            
            You will hear a beep if you choose incorrectly 
            or miss a response.
            
            Press 5 to hear the error sound
            """
            
        self.presentInstruction(intro_text)
        
        # show beeps
        error_sound.play()
        
        self.presentInstruction("Press 5 to hear the 'miss' sound")
        miss_sound.play(); core.wait(.5)
        
        self.presentInstruction(
            """
            Before beginning the 1st task, you will get some practice 
            seeing each image when it is UNROTATED as well as how it 
            looks when it IS rotated. 
            
            First, we will show you each of the 15 images in their normal 
            unrotated positions. Your job is simply to look at each image
            to become familiar with it.
            
            Press the left and right keys to move through the images.
            
            Press 5 to continue
            """)
        self.run_familiarization()
        
        
        self.presentInstruction(
            """
            Now we will show you each image next to its rotated version. 
            Your job is to pick out which of the two images IS NOT rotated 
            by pressing the LEFT and RIGHT keys on your keyboard.
            
            Press the direction where the image is UNROTATED.

            Wait for the experimenter
            """)
        
        self.run_familiarization_test()

                
        # structure learning 
        self.presentInstruction(
            """
            Finished with familiarization. In the next section, 
            indicate whether the image is unrotated or rotated.
            
            If UNROTATED, press the %s. 
            If the image is ROTATED press the %s
            
            Each image will only come up on the screen for a short 
            amount of time. Please respond as quickly and accurately 
            as possible.
            
            There will be three breaks.
            
            Wait for the experimenter
            """  % ((self.action_keys[0] + ' key').upper(), 
                    (self.action_keys[1] + ' key').upper()))
        
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

