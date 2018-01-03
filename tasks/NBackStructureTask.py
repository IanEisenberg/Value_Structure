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
from utils.utils import gen_nbackstructure_trials

# play a sound at beginning to ensure that psychopy's sound is working
error_sound = sound.Sound(secs=.1,value=500)
miss_sound = sound.Sound(secs=.1,value=700)

class NBackStructureTask(BaseExp):
    """ class defining a probabilistic context task
    """
    
    def __init__(self, expid, subjid, save_dir, stim_files, graph, 
                 trial_params = {}, fullscreen = False):
        # set up default trial params
        self.trial_params = {'N': 2,
                             'num_trials': 1400,
                             'num_practice_trials': 60,
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
        
        # define trial sequence
        self.practice_seed = self.trial_params['seed']+10 if self.trial_params['seed']  else None
        self.practice_trials = gen_nbackstructure_trials(
                                self.graph, 
                                self.stim_files, 
                                self.trial_params['num_practice_trials'], 
                                exp_stage='practice_structure_learning',
                                n=self.trial_params['N'],
                                seed=self.practice_seed)
        np.mean([i['nback_match'] for i in self.practice_trials])

        self.trials = gen_nbackstructure_trials(
                               self.graph, 
                               self.stim_files, 
                               self.trial_params['num_trials'], 
                               exp_stage='structure_learning',
                               n=self.trial_params['N'],
                               seed=self.trial_params['seed'])


        # set up static variables
        self.action_keys = ['z','m']
        np.random.shuffle(self.action_keys)
        self.n_value_ratings = 3
        # init Base Exp
        super(NBackStructureTask, self).__init__(expid, subjid, save_dir, fullscreen)
            
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
        trial['correct'] = np.nan
        trial['onset']=core.getTime() - self.startTime
        trial['response'] = np.nan
        trial['rt'] = np.nan
        trial['secondary_responses'] = []
        trial['secondary_rts'] = []
        trial['trialnum'] = self.trialnum
        # change nback_match to false if the last trial was a pause trial
        if len(self.structuredata)>0 and self.structuredata[-1]['exp_stage'] == 'break':
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
            choice = ['not_match','match'][self.action_keys.index(first_key[0])]
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
        #print('Nback_Match: %s, correct: %s' % (trial['nback_match'], trial['correct']))
        # log trial and add to data
        self.writeToLog(json.dumps(trial))
        self.structuredata.append(trial)
        return trial
                
    def run_familiarization(self):
        i=0
        stims = sample(self.stim_files, len(self.stim_files))
        while i<len(stims):
            filey = stims[i]
            keys = self.presentStim(filey, ['left','right'])
            if keys[0][0] == 'right':
                i+=1
            elif i>0:
                i-=1
    
    def run_graph_practice(self):
        self.trialnum = 0
        # start graph learning
        self.presentTextToWindow('Get Ready!', duration=2)
        self.clearWindow()
        acc = []
        for trial in self.practice_trials:
            out = self.presentTrial(trial)  
            acc.append(out['correct'])
        return np.mean(acc)
                
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
                self.presentInstruction("Press 5 to restart")
                break_length = clock.getTime()
                self.structuredata.append({'exp_stage': 'break',
                                           'duration': break_length})
                self.presentTextToWindow('Get Ready!', duration=2)
        
    def run_task(self):
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
            will be shown one at a time.
            
            Your task is to indicate whether the image shown is 
            the same as the one shown 2 images before by pressing 
            the corresponding key:
            
            %s key: Same as %s before
            %s key: Different than %s before
            
            You will hear a beep if you choose incorrectly 
            or miss a response.
            
            Press 5 to [blank]
            """
            
        self.presentInstruction(intro_text.replace('[blank]',
                                                   'hear the error beep') 
                                    % (self.action_keys[1].title(), 
                                        self.trial_params['N'],
                                        self.action_keys[0].title(),
                                        self.trial_params['N']))
        
        # show beeps
        error_sound.play()
        
        self.presentInstruction(intro_text.replace('[blank]', 
                                                   'hear the miss beep') 
                                    % (self.action_keys[1].title(), 
                                        self.trial_params['N'],
                                        self.action_keys[0].title(),
                                        self.trial_params['N']))
        miss_sound.play(); core.wait(.5)
        
        self.presentInstruction(
            """
            We will start by familiarizing you with the images
            
            Press the left and right keys to move through the images.
            
            Press 5 to continue...
            """)
        self.run_familiarization()

                
        # structure learning 
        self.presentInstruction(
            """
            Finished with familiarization. We will now practice responding
            to the images. Remember, indicate whether the image shown is
            the same as the one shown 2 images before by pressing:
            
                %s key: Same as %s before
                %s key: Different than %s before
            
            Each image will only come up on the screen for a short 
            amount of time. Please respond as quickly and accurately 
            as possible.

            Wait for the experimenter
            """ % (self.action_keys[1].title(), 
                    self.trial_params['N'],
                    self.action_keys[0].title(),
                    self.trial_params['N']))
        practice_over = False
        practice_repeats = 0
        while not practice_over:
            avg_acc = self.run_graph_practice()
            self.presentTextToWindow('Wait for Experimenter')
            keys, time = self.waitForKeypress([self.trigger_key, '0'])
            if keys[0][0] == self.trigger_key:
                practice_over = True
            else:
                practice_repeats += 1
                self.practice_trials = gen_nbackstructure_trials(
                                self.graph, 
                                self.stim_files, 
                                self.trial_params['num_practice_trials'], 
                                exp_stage='practice_structure_learning',
                                n=self.trial_params['N'],
                                seed=self.practice_seed+practice_repeats)
        
        self.presentInstruction(
            """
            Done with practice. We will now start the first task which will
            take roughly 35 minutes. There will be 3 breaks.
            
                %s key: Same as %s before
                %s key: Different than %s before
            
            
            Wait for the experimenter
            """ % (self.action_keys[1].title(), 
                    self.trial_params['N'],
                    self.action_keys[0].title(),
                    self.trial_params['N']))
            
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
        self.closeWindow()

