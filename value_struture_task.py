#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 08:56:52 2017

@author: ian
"""

"""
generic task using psychopy
"""

import cPickle
import datetime
import json
import numpy as np
import pandas as pd
from psychopy import visual, core, event, sound
import sys,os
import yaml

class valueStructure:
    """ class defining a probabilistic context task
    """
    
    def __init__(self,subjid,save_dir,trials,
                 familiarization_trials,
                 fullscreen = False):
        # set up "holder" variables
        self.alldata=[]  
        self.pointtracker=0
        self.startTime=[]
        self.textStim=[]
        self.trialnum = 0
        
        # set up argument variables
        self.familiarization_trials = familiarization_trials
        self.fullscreen = fullscreen
        self.save_dir = save_dir  
        self.subjid=subjid
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.trials = trials
        self.stim_files = list(np.unique([t['stim_file'] for t in self.trials]))
        
        # set up static keys
        self.action_keys = ['left','right']
        self.quit_key = 'q'
        self.trigger_key = '5'
        self.text_color = [1]*3
        
        # set up window
        self.win=[]
        self.window_dims=[800,600]
        
        # set up recording files
        self.logfilename='%s_%s.log'%(self.subjid,self.timestamp)
        self.datafilename='%s_%s.pkl'%(self.subjid,self.timestamp)
        # log initial state
        try:
            self.writeToLog(self.toJSON())
        except IOError:
            os.makedirs(os.path.join(self.save_dir,'Log'))
            self.writeToLog(self.toJSON())
            
    #**************************************************************************
    # ******* Function to Save Data **************
    #**************************************************************************
    
    def toJSON(self):
        """ log the initial conditions for the task. Exclude the list of all
        trials (stimulusinfo), the bot, and taskinfo (self.__dict__ includes 
        all of the same information as taskinfo)
        """
        init_dict = {k:self.__dict__[k] for k in self.__dict__.iterkeys() if k 
                    not in ('clock', 'stimulusInfo', 'alldata', 'bot', 'taskinfo','win')}
        return json.dumps(init_dict)
    
    def writeToLog(self,msg):
        f=open(os.path.join(self.save_dir,'Log',self.logfilename),'a')
        f.write(msg)
        f.write('\n')
        f.close()
         
    def writeData(self):
        save_loc = os.path.join(self.save_dir,'RawData',self.datafilename)
        data = {}
        data['subcode']=self.subjid
        data['timestamp']=self.timestamp
        data['taskdata']=self.alldata
        try:
            f=open(save_loc,'w')
        except IOError:
            os.makedirs(os.path.split(save_loc)[0])
            f=open(save_loc,'w')
        cPickle.dump(data,f)
    
    #**************************************************************************
    # ******* Display Functions **************
    #**************************************************************************
    
    def setupWindow(self):
        """ set up the main window
        """
        self.win = visual.Window(self.window_dims, allowGUI=False, 
                                 fullscr=self.fullscreen, monitor='testMonitor', 
                                 units='norm', allowStencil=True,
                                 color=[-1,-1,-1])   

        self.win.flip()
        self.win.flip()
        
        
    def presentTextToWindow(self,text,size=.15, duration=None):
        """ present a text message to the screen
        return:  time of completion
        """
        
        if not self.textStim:
            self.textStim=visual.TextStim(self.win, text=text,font='BiauKai',
                                height=size,color=self.text_color, colorSpace=u'rgb',
                                opacity=1,depth=0.0,
                                alignHoriz='center',
                                alignVert='center',
                                wrapWidth=50)
        else:
            self.textStim.setText(text)
            self.textStim.setHeight(size)
            self.textStim.setColor(self.text_color)
        self.textStim.draw()
        self.win.flip()
        if duration:
            core.wait(duration)
        return core.getTime()

    def clearWindow(self):
        """ clear the main window
        """
        if self.textStim:
            self.textStim.setText('')
            self.win.flip()
        else:
            self.presentTextToWindow('')

    def waitForKeypress(self,key=[]):
        """ wait for a keypress and return the pressed key
        - this is primarily for waiting to start a task
        - use getResponse to get responses on a task
        """
        start=False
        event.clearEvents()
        while start==False:
            key_response=event.getKeys()
            if len(key_response)>0:
                if key:
                    if key in key_response or self.quit_key in key_response:
                        start=True
                else:
                    start=True
        self.clearWindow()
        return key_response,core.getTime()
        
    def closeWindow(self):
        """ close the main window
        """
        if self.win:
            self.win.close()

    def checkRespForQuitKey(self,resp):
        if self.quit_key in resp:
            self.shutDownEarly()

    def shutDownEarly(self):
        self.closeWindow()
        sys.exit()

    def presentInstruction(self, text):
        self.presentTextToWindow(text)
        resp,self.startTime=self.waitForKeypress(self.trigger_key)
        self.checkRespForQuitKey(resp)
        event.clearEvents()
        
    def presentStim(self, stim_file, rotation, textstim=None, duration=None):
        stim = visual.ImageStim(self.win, 
                                image=stim_file,
                                ori=rotation,
                                units='deg')
        stim.draw()
        if textstim:
            textstim.draw()
        
        self.win.flip()
        
        recorded_keys = []
        stim_clock = core.Clock()
        if duration:
            while stim_clock.getTime() < duration:
                keys = event.getKeys(self.action_keys + [self.quit_key],
                                     timeStamped=True)
                for key,response_time in keys:
                    self.checkRespForQuitKey(key)
                    recorded_keys+=keys
        else:
            while len(recorded_keys) == 0:
                keys = event.getKeys(self.action_keys + [self.quit_key],
                                     timeStamped=True)
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
        trial['onset']=core.getTime() - self.startTime
        trial['response'] = 999
        trial['rt'] = 999
        trial['correct'] = False
        # present stimulus and get response
        event.clearEvents()
        trialClock.reset()
        keys = self.presentStim(trial['stim_file'], 
                                trial['rotation'],
                                trial['duration'])
        if len(keys)>0:
            first_key = keys[0]
            choice = ['unrot','rot'][self.action_keys.index(first_key[0])]
            print('Choice: %s' % choice)
            # record response
            trial['response'] = choice
            trial['rt'] = first_key[1]
            # record any responses after the first
            trial['secondary_responses']=[i[0] for i in keys[1:]]
            trial['secondary_rts']=[i[1] for i in keys[1:]]
            # get feedback and update tracker
            correct_choice = ['unrot','rot'][[0,90].index(trial['rotation'])]
            if correct_choice == choice:
                trial['correct']=True
                # record points for bonus
                self.pointtracker += 1
            else:
                error_sound = sound.Sound(secs=.1,value=500)
                error_sound.play() 
        # If subject did not respond within the stimulus window clear the stim
        # and admonish the subject
        if trial['rt']==999:
            miss_sound = sound.Sound(secs=.1,value=1000)
            miss_sound.play() 
            
        # log trial and add to data
        self.writeToLog(json.dumps(trial))
        self.alldata.append(trial)
        return trial
    
    
    def run_familiarization(self):
        i=0
        while i<15:
            filey = self.stim_files[i//2]
            text = ['Unrotated','Rotated'][i%2==0]
            textstim = visual.TextStim(self.win, text, pos=[0,.5], units='norm')
            keys = self.presentStim(filey, [0,90][i%2==0], textstim)
            if keys[0][0] == 'right':
                i+=1
            elif i>0:
                i-=1
                
    def run_familiarization_test(self):
        np.random.shuffle(self.familiarization_trials)
        for trial in self.familiarization_trials:
            self.presentTrial(trial)
                            
    def run_graph_learning(self):
        self.presentTextToWindow('Get Ready!', duration=2)
        self.clearWindow()
        for trial in self.trials:
            self.presentTrial(trial)
            
    def run_task(self, pause_trials = None):
        self.setupWindow()
        # instructions
        self.presentInstruction(
            """
            In the first part of this study,
            stimuli will be shown one at a time for a short amount of time. Your
            task is to indicated whether the stimulus is rotated or unrotated.
            
                
            We will start by familiarizing you with the stimuli. 
            You will see each stimulus twice, first unrotated, then rotated. 
            """)


        learned=False
        while not learned:
            self.run_familiarization()
            self.presentInstruction(
                """
                We will now practice responding to the stimuli. Indicate whether
                the stimulus is unrotated or rotated.
                
                        Left Key: Unrotated
                        Right Key: Rotated
                """)
            self.run_familiarization_test()
            acc = np.mean([t['correct'] for t in self.alldata 
                           if t['exp_stage'] == 'familiarization_test'])
            print(acc)
            if acc>.8:
                learned=True
            else:
                self.presentInstruction(
                    """
                    Seems you can use a refresher! Please look over the
                    stimuli again and try to remember which way the stimulus
                    is unrotated
                    """)
                
        self.presentInstruction(
            """
            Finished with familiarization. In the next section, indicated
            whether the image is unrotated or rotated.
            
                Left Key: Unrotated
                Right Key: Rotated
                
            Each correct choice will earn you $1.  
            """)
        self.run_graph_learning()
                
        # clean up and save
        self.writeData()
        self.presentTextToWindow('Thank you. Please wait for the experimenter',
                                 size=.05)
        self.waitForKeypress(self.quit_key)
        self.closeWindow()


