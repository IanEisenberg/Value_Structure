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
from psychopy import visual, core, event
import sys,os
import yaml

class valueStructure:
    """ class defining a probabilistic context task
    """
    
    def __init__(self,subjid,save_dir,trials,fullscreen = False):
        # set up "holder" variables
        self.alldata=[]  
        self.pointtracker=0
        self.startTime=[]
        self.textStim=[]
        self.trialnum = 0
        
        # set up argument variables
        self.fullscreen = fullscreen
        self.save_dir = save_dir  
        self.subjid=subjid
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.trials = trials
        
        # set up static keys
        self.action_keys = ['left','right']
        self.quit_key = 'q'
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
        
        
    def presentTextToWindow(self,text,size=.15):
        """ present a text message to the screen
        return:  time of completion
        """
        
        if not self.textStim:
            self.textStim=visual.TextStim(self.win, text=text,font='BiauKai',
                                height=size,color=self.text_color, colorSpace=u'rgb',
                                opacity=1,depth=0.0,
                                alignHoriz='center',wrapWidth=50)
        else:
            self.textStim.setText(text)
            self.textStim.setHeight(size)
            self.textStim.setColor(self.text_color)
        self.textStim.draw()
        self.win.flip()
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

    def presentStim(self, stim_file, rotation, duration):
        stim = visual.ImageStim(self.win, 
                                image=stim_file,
                                ori=rotation,
                                units='deg')
        stim.draw()
        self.win.flip()
        
        recorded_keys = []
        stim_clock = core.Clock()
        while stim_clock.getTime() < duration:
            keys = event.getKeys(self.action_keys + [self.quit_key],
                                 timeStamped=True)
            for key,response_time in keys:
                # check for quit key
                if key == self.quit_key:
                    self.shutDownEarly()
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
            choice = ['not_rot','rot'][self.action_keys.index(first_key[0])]
            print('Choice: %s' % choice)
            # record response
            trial['response'] = choice
            trial['rt'] = first_key[1]
            # record any responses after the first
            trial['secondary_responses']=[i[0] for i in keys[1:]]
            trial['secondary_rts']=[i[1] for i in keys[1:]]
            # get feedback and update tracker
            correct_choice = 'rot'
            if correct_choice == choice:
                trial['correct']=True
                # record points for bonus
                self.pointtracker += 1
        # If subject did not respond within the stimulus window clear the stim
        # and admonish the subject
        if trial['rt']==999:
            self.clearWindow()            
            core.wait(1)
            self.presentTextToWindow('Please Respond Faster')
            core.wait(1)
            self.clearWindow()
            
        # log trial and add to data
        self.writeToLog(json.dumps(trial))
        self.alldata.append(trial)
        return trial
            
    def run_task(self, pause_trials = None):
        self.setupWindow()
        self.startTime = core.getTime()
        
        for trial in self.trials:
            self.presentTrial(trial)
                
        # clean up and save
        self.writeData()
        self.presentTextToWindow('Thank you. Please wait for the experimenter',
                                 size=.05)
        self.waitForKeypress(self.quit_key)
        self.closeWindow()



# shuffle stims
stims = ['images/%s.png' % '9' for i in range(1,16)]
np.random.shuffle(stims)
# graph structure
graph = {0: [1,2,3,14],
         1: [0,2,3,4],
         2: [0,1,3,4],
         3: [0,1,2,4],
         4: [1,2,3,5],
         5: [4,6,7,8],
         6: [5,7,8,9],
         7: [5,6,8,9],
         8: [5,6,7,9],
         9: [6,7,8,10],
         10: [9,11,12,13],
         11: [10,12,13,14],
         12: [10,11,13,14],
         13: [10,11,12,14],
         14: [11,12,13,0]}

def gen_trials(trial_count=100):
    trials = []
    curr_i = np.random.randint(0,14)
    for i in range(trial_count):
        trial = {'stim_index': curr_i,
                 'stim_file': stims[curr_i],
                 'duration': .5,
                 'rotation': 90*np.random.choice([0,1])}
        trials.append(trial)
        # random walk
        curr_i = np.random.choice(graph[curr_i])
    return trials
    
subj = 'test'
save_dir = os.path.join('Data',subj)
trials = [{'duration':2}]*4
task = valueStructure(subj, save_dir, gen_trials())
task.run_task()