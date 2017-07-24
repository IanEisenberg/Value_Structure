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
from psychopy import visual, core, event, sound
from random import sample
import sys,os

class valueStructure:
    """ class defining a probabilistic context task
    """
    
    def __init__(self,subjid, save_dir, stim_files, graph,
                 trials, familiarization_trials, fullscreen = False):
        # set up "holder" variables
        self.valuedata=[]
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
        np.random.shuffle(self.action_keys)
        self.quit_key = 'q'
        self.node_values = np.round(sample(np.linspace(0,10,15),15),1)
        self.labeled_nodes = [0,1,10,11]
        self.node_values[self.labeled_nodes] = [9.3,8,2.5,1.2]
        self.n_value_ratings = 12
        self.trigger_key = '5'
        self.test_familiarization = False
        self.text_color = [1]*3
        
        # set up window
        self.win=[]
        self.window_dims=[1920,1080]
        
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
                    not in ('clock', 'stimulusInfo', 
                            'structuredata', 'bot', 'taskinfo','win',
                            'node_values')}
        return json.dumps(init_dict)
    
    def writeToLog(self,msg):
        f=open(os.path.join(self.save_dir,'Log',self.logfilename),'a')
        f.write(msg)
        f.write('\n')
        f.close()
         
    def writeData(self):
        # create taskdata object
        taskdata = {
                    'graph': self.graph,
                    'action_keys': self.action_keys,
                    'labeled_nodes': self.labeled_nodes,
                    'node_values': list(self.node_values)
                    }
        # save data
        save_loc = os.path.join(self.save_dir,'RawData',self.datafilename)
        data = {}
        data['subcode']=self.subjid
        data['taskdata'] = taskdata
        data['timestamp']=self.timestamp
        data['structuredata']=self.structuredata
        data['valuedata']=self.valuedata
        try:
            f=open(save_loc,'wb')
        except IOError:
            os.makedirs(os.path.split(save_loc)[0])
            f=open(save_loc,'wb')
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
        
        # set up ratio for squares/cricles
        stim_ratio = float(self.win.size[0])/self.win.size[1]
        self.stim_size = np.array([.3, stim_ratio*.3])
        
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
    
    def get_labeled_banner(self, labeled_stims, positions, height, value=True):
        banner = []
        labeled_stims = sample(labeled_stims, len(labeled_stims))
        for i, labeled_stim in enumerate(labeled_stims):
            stim_file, value = labeled_stim
            # logo
            stim = visual.ImageStim(self.win, image=stim_file,
                                units='norm', 
                                pos=(positions[i],height),
                                size=self.stim_size*.6)
            banner.append(stim)
            if value:
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
        for i, labeled_stim in enumerate(labeled_stims):
            stim_file, value = labeled_stim
            # logo
            stim_pos = float(value-limits[0])/(limits[1]-limits[0])
            stim_pos = stim_pos*pos_limits[1]+(1-stim_pos)*pos_limits[0]
            stim = visual.ImageStim(self.win, image=stim_file,
                                units='norm', 
                                pos=(stim_pos,
                                     scale.pos[1]+.16),
                                size=self.stim_size*.3)
            banner.append(stim)
        return banner
    
    def presentInstruction(self, text):
        self.presentTextToWindow(text, size = .07)
        resp,self.startTime=self.waitForKeypress(self.trigger_key)
        self.checkRespForQuitKey(resp)
        event.clearEvents()
        
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
        trial['onset']=core.getTime() - self.startTime
        trial['response'] = -1
        trial['rt'] = -1
        trial['secondary_responses'] = []
        trial['secondary_rts'] = []
        trial['correct'] = False
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
        
    def run_value_rating(self, trials, labeled_stims=None):
        for trial in trials:
            labeled_points=None
            ratingScale = visual.RatingScale(self.win, low=0, high=10,
                                             precision=10,
                                             scale='',
                                             labels=('0','5','10'),
                                             stretch=2,
                                             pos=(0,-.5),
                                             markerColor='white')
            if labeled_stims:
                labeled_points = self.place_labeled_stims(labeled_stims,
                                                          ratingScale)
        
            stim = trial.pop('stim')
            while ratingScale.noResponse:
                stim.draw()
                ratingScale.draw()
                if labeled_points:
                    for p in labeled_points:
                        p.draw()
                        
                self.win.flip()
            trial['rating'] = ratingScale.getRating()
            trial['rt'] = ratingScale.getRT()
            trial['history'] = ratingScale.getHistory()
            self.valuedata.append(trial)
            self.writeToLog(json.dumps(trial))
        
    def run_task(self, pause_trials = None):
        self.setupWindow()
        
        self.presentInstruction('Welcome! Press 5 to continue...')
        
        # instructions
        self.presentInstruction(
            """
            In the first part of this study, stimuli
            will be shown one at a time for a short 
            amount of time. Each stimulus is associated
            with a different value between 0 RMB and 10 RMB.
            
            Your first task is to indicated whether 
            each stimulus is rotated or unrotated.
            
            We will start by familiarizing you with the stimuli. 
            Press the left and right keys to move through the stimuli.
            
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
                    
                            %s Key: Unrotated
                            %s Key: Rotated
                            
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
                        logos again and try to remember which way the stimulus
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
            
                %s Key: Unrotated
                %s Key: Rotated
            
            Each logo will only come up on the screen for a short 
            amount of time. Please respond as quickly and accurately 
            as possible.
            
            You will hear a beep if you choose incorrectly.
            This section takes a long time, so there will be two
            breaks.
            
            Press 5 to continue...
            """ % (self.action_keys[0], self.action_keys[1]))
        self.run_graph_learning()
        
        self.presentInstruction(
            """
            Finished with that section. Take a break!
            
            In the next section we will ask you to bid on the
            different stimuli (we will explain how to bid
            on the next screen). First, we will tell you the 
            value of 4 of the stimuli.
            
            When you are ready, press 5 to continue...
            """)
        
        # labeling phase
        label_instruction = visual.TextStim(self.win, 
                                            "Above are the values of 4 stimuli",
                                            pos=[0,-.3], units='norm')
        labeled_stims = [(self.stim_files[i],self.node_values[i]) 
                         for i in self.labeled_nodes]
        labeled_banner = self.get_labeled_banner(labeled_stims,
                                                 [-.6,-.2,.2,.6], .4)
        for c in labeled_banner:
            c.draw()
        label_instruction.draw()
        self.win.flip()
        self.waitForKeypress()
        
        self.presentInstruction(
            """
            We will now start the bidding. You will start with 10 RMB,
            which you can use to bid. Remember that each stimulus is 
            associated with a value between 0 RMB and 10 RMB.
            
            Bidding works by stating a value between 0 RMB
            and 10 RMB that you are willing to pay for a stimulus. 
            At the end of the study, one trial will be
            randomly chosen. Once chosen, a random number between
            0-10 will be chosen. If that number is above your bid,
            you will not pay for the stimulus and just keep 10 RMB.
            
            However, if the number drawn is below your bid, you will pay
            that drawn amount (not your original bid), and also get the
            value of the stimulus.
            
            At the end of the experiment you will be paid the
            combination of your original 10 RMB and the results of the 
            bid. Thus you can potentially leave with 0-20 RMB
            
            Press 5 to continue...
            """)
        
        # value rating phase
        unknown_stims = []
        rating_stims = [self.stim_files[i] for i in [2,4,5,7,9,12,14]]
        for rep in range(self.n_value_ratings):
            unknown_stims+=sample(rating_stims,len(rating_stims))
        rating_trials = []
        for stim_file in unknown_stims:
            stim_i = self.stim_files.index(stim_file)
            stim = visual.ImageStim(self.win, image=stim_file,
                                units='norm', pos=(0,.2),
                                size=self.stim_size)
            trial = {'stim': stim,
                     'stim_file': stim_file,
                     'stim_index': stim_i,
                     'exp_stage': 'value_rating'}
            rating_trials.append(trial)
        self.run_value_rating(rating_trials, labeled_stims)
        
        # randomly choose a trial
        bid_won = False
        total_win = 10
        selected_bid = np.random.choice(self.valuedata)
        random_price = np.random.rand()*10
        if random_price < selected_bid['rating']:
            bid_won = True
            stim_value = self.node_values[selected_bid['stim_index']]
            total_win = total_win - round(random_price,1) \
                        + stim_value
        if bid_won == True:
            self.presentInstruction(
                """
                On the random trial we drew you bid %s RMB. The random price
                drawn was %s RMB, so you won the bid. You paid the random price 
                for a stimulus worth %s. Your total earning is %s RMB
                """ % (selected_bid['rating'], round(random_price,1),
                        stim_value, total_win)
                )
        else:
            self.presentInstruction(
                    """
                    On the random trial we drew you bid %s. The random price
                    drawn was %s, so you didn't win the bid. Thus you won
                    10 RMB.
                    """ % (selected_bid['rating'], round(random_price,1))
                    )
        
        
        # clean up and save
        self.writeData()
        self.presentTextToWindow('Thank you. Please wait for the experimenter',
                                 size=.1)
        self.waitForKeypress(self.quit_key)
        self.closeWindow()
        return total_win

