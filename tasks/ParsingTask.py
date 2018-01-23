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
from utils.utils import gen_parsing_trials

# play a sound at beginning to ensure that psychopy's sound is working
error_sound = sound.Sound(secs=.1,value=500)
miss_sound = sound.Sound(secs=.1,value=700)

class ParsingTask(BaseExp):
    """ class defining a probabilistic context task
    """
    
    def __init__(self, expid, subjid, save_dir, stim_files, graph, 
                 trial_params = {}, fullscreen = False):
        # set up default trial params
        self.trial_params = {'num_trials': 1400,
                             'seed': None}
        
        # set up "holder" variables
        self.parsedata = []  
        self.pointtracker = 0
        self.trialnum = 0
        self.startTime = []
        
        # set up argument variables
        self.graph = graph
        self.stim_files = stim_files
        self.trial_params.update(trial_params)
        
        # define trial sequence

        self.trials = gen_parsing_trials(
                               self.graph, 
                               self.stim_files, 
                               self.trial_params['num_trials'], 
                               seed=self.trial_params['seed'])


        # set up static variables
        self.num_breaks = 2
        self.action_keys = ['space']
        # init Base Exp
        super(ParsingTask, self).__init__(expid, subjid, save_dir, fullscreen)
            
    #**************************************************************************
    # ******* Display Functions **************
    #**************************************************************************

    def presentStim(self, stim_file, allowed_keys=None, textstim=None, 
                    duration=None, correct_choice=None):
        if allowed_keys is None:
            allowed_keys = self.action_keys
        size = self.stim_size
        # present stim
        stim = visual.ImageStim(self.win, 
                                image=stim_file,
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
        trial['correct'] = False
        trial['onset']=core.getTime() - self.startTime
        trial['response'] = np.nan
        trial['rt'] = np.nan
        trial['secondary_responses'] = []
        trial['secondary_rts'] = []
        trial['trialnum'] = self.trialnum
        # change nback_match to false if the last trial was a pause trial
        if len(self.parsedata)>0 and self.parsedata[-1]['exp_stage'] == 'break':
            trial['nback_match'] = 0
        correct_choice = self.action_keys[trial['nback_match']]
        # present stimulus and get response
        event.clearEvents()
        trialClock.reset()
        keys = self.presentStim(trial['stim_file'], 
                                duration = trial['duration'],
                                correct_choice=correct_choice)
        if len(keys)>0:
            first_key = keys[0]
            # record response
            trial['response'] = 'break'
            trial['rt'] = first_key[1]
            # record any responses after the first
            trial['secondary_responses']=[i[0] for i in keys[1:]]
            trial['secondary_rts']=[i[1] for i in keys[1:]]
            if trial['community_transition']:
                trial['correct']=True
        else:
            if trial['community_transition'] == 0:
                trial['correct'] = True

        #print('Nback_Match: %s, correct: %s' % (trial['nback_match'], trial['correct']))
        # log trial and add to data
        self.writeToLog(json.dumps(trial))
        self.parsedata.append(trial)
        return trial
                
    def run_parsing(self):
        self.trialnum = 0
        # start graph learning
        pause_trials = [len(self.trials)//self.num_breaks*i for i in range(1,self.num_breaks)]
        timer_text = "Take a break!\n\nContinue in: \n\n       "
        self.presentTextToWindow('Get Ready!', duration=2)
        self.clearWindow()
        for trial in self.trials:
            self.presentTrial(trial)
            if self.trialnum in pause_trials:
                clock = core.Clock()
                self.presentTimer(duration=30, text=timer_text)
                self.presentInstruction("Press 5 to continue")
                break_length = clock.getTime()
                self.parsedata.append({'exp_stage': 'break',
                                           'duration': break_length})
                self.presentTextToWindow('Get Ready!', duration=2)
        
    def run_task(self):
        self.setupWindow()
        self.stim_size = self.getSquareSize(self.win)
        self.startTime = core.getTime()
        # instructions
        intro_text = """
            In this section, you’ll see a stream of the same 
            images. We want you to press the SPACEBAR at times in the 
            sequence that you feel are natural breaking points. 
            
            If you are not sure, go with your gut feeling. 
            
            Try to make your responses as quickly and 
            accurately as possible
            """
            
        self.presentInstruction(intro_text)
    
        self.run_parsing()
        
        # clean up and save
        taskdata = {
                'stim_files': self.stim_files,
                'graph': self.graph
                } 
        otherdata = {'structuredata': self.parsedata}
        self.writeData(taskdata, otherdata)
        
        self.presentInstruction(
            """
            Done with the first task! Take a break for a few minutes.
            
            Please wait for the experimenter.
            """)
        self.closeWindow()

