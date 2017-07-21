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

class valueStructure:
    """ class defining a probabilistic context task
    """
    
    def __init__(self,subjid, save_dir, stim_files, graph,
                 trials, familiarization_trials, fullscreen = False):
        # set up "holder" variables
        self.pricedata=[]
        self.structuredata=[]  
        self.pointtracker=0
        self.startTime=[]
        self.textStim=[]
        self.trialnum = 0
        
        # set up argument variables
        self.familiarization_trials = familiarization_trials
        self.fullscreen = fullscreen
        self.graph = graph
        self.save_dir = save_dir  
        self.stim_files = stim_files
        self.subjid=subjid
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.trials = trials
        
        # set up static variables
        self.action_keys = ['left','right']
        self.quit_key = 'q'
        self.labeled_nodes = [(1,5), (2,3), (8,10), (9,1)] #node: price
        self.n_price_ratings = 20
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
                    not in ('clock', 'stimulusInfo', 'structuredata', 'bot', 'taskinfo','win')}
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
        data['taskdata']=self.structuredata
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
    
    def get_labeled_banner(self, labeled_stims):
        positions = [-.6,-.2,.2,.6]
        prices = [i[1] for i in self.labeled_nodes]
        banner = []
        for i, stim_file in enumerate(labeled_stims):
            # logo
            stim = visual.ImageStim(self.win, image=stim_file,
                                units='norm', 
                                pos=(positions[i],.7))
            banner.append(stim)
            # price
            pricestim = visual.TextStim(self.win, '$%s' % prices[i], 
                                       pos=(positions[i],.4), units='norm')
            banner.append(pricestim)
        return banner
            
            
    def presentInstruction(self, text):
        self.presentTextToWindow(text, size = .1)
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
                                duration = trial['duration'])
        if len(keys)>0:
            first_key = keys[0]
            choice = ['unrot','rot'][self.action_keys.index(first_key[0])]
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
        self.structuredata.append(trial)
        return trial
    
    
    def run_familiarization(self):
        i=0
        while i<len(self.stim_files)*2:
            filey = self.stim_files[i//2]
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
        self.presentTextToWindow('Get Ready!', duration=2)
        self.clearWindow()
        for trial in self.trials:
            self.presentTrial(trial)
    
    def run_price_rating(self, trials, context=None):
        for trial in trials:
            ratingScale = visual.RatingScale(self.win, low=0, high=10,
                                             precision=10,
                                             scale='Cost in RMB',
                                             labels=('0','5','10'),
                                             stretch=1.5,
                                             pos=(0,-.6))
            while ratingScale.noResponse:
                if context:
                    for c in context:
                        c.draw()
                trial['stim'].draw()
                ratingScale.draw()
                self.win.flip()
            initial_rating = {}
            initial_rating['rating'] = ratingScale.getRating()
            initial_rating['rt'] = ratingScale.getRT()
            initial_rating['history'] = ratingScale.getHistory()
            self.pricedata.append(initial_rating)
        
    def run_task(self, pause_trials = None):
        self.setupWindow()
        
        self.presentInstruction('Welcome! Press 5 to continue...')
        #initial price guess
        initial_question = "How much does a non-alcoholic bottled drink cost on average?"
        textstim = visual.TextStim(self.win, initial_question, 
                                   pos=[0,.5], units='norm')
        self.run_price_rating([{'stim': textstim}])
        
        # instructions
        self.presentInstruction(
            """
            In the first part of this study,
            logos for different drinks will be shown one at a time for 
            a short amount of time. Your task is to indicated 
            whether the logo is rotated or unrotated.
            
                
            We will start by familiarizing you with the logos. 
            Press the left and right keys to move through the logos.
            
            Press 5 to continue...
            """)

        learned=False
        while not learned:
            self.run_familiarization()
            self.presentInstruction(
                """
                We will now practice responding to the stimuli. 
                Indicate whether the stimulus 
                is unrotated or rotated.
                
                        Left Key: Unrotated
                        Right Key: Rotated
                        
                Press 5 to continue...
                """)
            self.run_familiarization_test()
            acc = np.mean([t['correct'] for t in self.structuredata 
                           if t['exp_stage'] == 'familiarization_test'])
            print(acc)
            if acc>.3:
                learned=True
            else:
                self.presentInstruction(
                    """
                    Seems you could use a refresher! Please look over the
                    logos again and try to remember which way the stimulus
                    is unrotated
                    
                    Press 5 to continue...
                    """)
                
        # structure learning 
        self.presentInstruction(
            """
            Finished with familiarization. In the next section, indicated
            whether the image is unrotated or rotated.
            
                Left Key: Unrotated
                Right Key: Rotated
                            
            Press 5 to continue...
            """)
        self.run_graph_learning()
        
        self.presentInstruction(
            """
            Finished with that section. Take a break!
            
            In the next section we will ask you to guess the price of the
            different drinks. We will tell you the price of 4 different drinks
            (represented by the logos) first.
            
            When you are ready, press 5 to continue...
            """)
        
        # labeling phase
        label_instruction = visual.TextStim(self.win, 
                                            "Above are the prices of 4 drinks",
                                            pos=[0,-.1], units='norm')
        labeled_stims = [self.stim_files[i[0]] for i in self.labeled_nodes]
        labeled_banner = self.get_labeled_banner(labeled_stims)
        for c in labeled_banner:
            c.draw()
        label_instruction.draw()
        self.win.flip()
        self.waitForKeypress()
        
        self.presentInstruction(
            """
            Now you will see the remaining logos. Please indicate the price
            you think the drink associated with each logo costs in the market.
            You will rate each logo multiple times.
            
            At the end of the experiment, if the average of your
            estimates for one drink is less
            than 1 RMB from the true value, you will earn 10 RMB.
            
            Press 5 to continue...
            """)
        
        # price rating phase
        unknown_stims = [s for s in self.stim_files 
                         if s not in labeled_stims]*self.n_price_ratings
        np.random.shuffle(unknown_stims)
        rating_trials = []
        for stim_file in unknown_stims:
            stim_file = stim = visual.ImageStim(self.win, image=stim_file,
                                units='norm', pos=(0,0))
            trial = {'stim': stim}
            rating_trials.append(trial)
        self.run_price_rating(rating_trials, labeled_banner)
        
        # clean up and save
        self.writeData()
        self.presentTextToWindow('Thank you. Please wait for the experimenter',
                                 size=.05)
        self.waitForKeypress(self.quit_key)
        self.closeWindow()


