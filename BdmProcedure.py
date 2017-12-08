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
from psychopy import visual
from random import sample

class BdmProcedure(BaseExp):
    """ class defining a probabilistic context task
    """
    
    def __init__(self,subjid, save_dir, stim_files, values,
                 labeled_nodes, fullscreen = False):
        super(BdmProcedure, self).__init__(subjid, save_dir, fullscreen)
        # set up "holder" variables
        self.valuedata = []
        self.pointtracker = 0
        self.startTime = []
        self.total_win = 0
        self.trialnum = 0
        
        
        # set up static variables
        self.action_keys = ['left','right']
        np.random.shuffle(self.action_keys)
        self.node_values = values
        self.labeled_nodes = labeled_nodes
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

    def run_bdm(self, value_trials):
        bid_won = False
        total_win = 10
        selected_bid = np.random.choice(value_trials)
        random_price = np.random.rand()*10
        if random_price < selected_bid['rating']:
            bid_won = True
            stim_value = self.node_values[selected_bid['stim_index']]
            total_win = total_win - round(random_price,1) \
                        + stim_value
        stim = visual.ImageStim(self.win, image=selected_bid['stim_file'],
                                units='norm', pos=(0,.6),
                                size=self.stim_size)
        stim.draw()
        if bid_won == True:
            self.presentInstruction(
            """
            On the random trial we drew you bid %s RMB on the
            stimulus above. 
            
            The random price drawn was %s RMB, so you won the bid. 
            The stimulus was worth %s. Your total earning is %s RMB
            
            Press 5 to continue...
            """ % (selected_bid['rating'], round(random_price,1),
                    round(stim_value,1), round(total_win,1))
                )
        else:
            self.presentInstruction(
                """
                On the random trial we drew you bid %s on the sitmulus
                above. 
                
                The random price drawn was %s, so you didn't win the bid. 
                Thus you won 10 RMB.
                
                Press 5 to continue...
                """ % (selected_bid['rating'], round(random_price,1))
                    )
        return total_win
            
    def run_bdm_explanation(self):
        trial_types = ['win','loss']*2
        first_instruction = visual.TextStim(self.win, 
            """
            Let's practice this procedure. Pretend you are
            bidding on the stimulus on the left, which is
            worth between 0 and 10 RMB.
            
            The amount you bid will determine the chance that 
            you win the bid. If you bid a lot, you will 
            likely get the value associated with the stimulus, 
            but you might pay a lot (remember, you pay the 
            random number drawn, not your bid!)
            
            However, if you bid very little, you won't pay much
            if you win, but you probably won't win the stimulus.
            
            Try bidding now!
            """, pos=[0,.35], units='norm',  height=.06)
            
        other_instructions = visual.TextStim(self.win, 
            """
            Let's try another trial. Please
            bid on how much you would bid
            for the stimulus on the left.
            
            Try bidding now!
            """, pos=[0,.35], units='norm',  height=.06)
        instruction = first_instruction
        for i, trial in enumerate(trial_types):
            bid_won = False
            stim = visual.ImageStim(self.win, 
                                    image='images/instruction_stim%s.png' % (i+1),
                                    units='norm', pos=(-.6,.2),
                                    size=self.stim_size)
            ratingScale = visual.RatingScale(self.win, low=0, high=10,
                                                 precision=10,
                                                 scale='Select your bid in RMB',
                                                 labels=('0','5','10'),
                                                 stretch=2,
                                                 pos=(0,-.5),
                                                 markerColor='white')
            
            ratingScale.draw()
            while ratingScale.noResponse:
                instruction.draw()
                ratingScale.draw()
                stim.draw()
                self.win.flip()
            rating = ratingScale.getRating()
            
            if trial=='win':
                random_price = np.random.rand()*rating
            else:
                random_price = rating+np.random.rand()*(10-rating)
            if random_price < rating:
                bid_won = True
            if bid_won == True:
                self.presentInstruction(
                """
                You bid %s RMB. The random price drawn was %s RMB, 
                so you won the bid. If this was a real bid,
                you'd pay %s and win the X RMB associated 
                with the stimulus.
                
                Press 5 to continue...
                """ % (rating, round(random_price,1),
                        round(random_price,1))
                    )
            else:
                self.presentInstruction(
                """
                You bid %s RMB. The random price drawn was %s RMB, 
                so you lost the bid. You pay nothing and get nothing.
                
                Press 5 to continue...
                """ % (rating, round(random_price,1))
                    )
            instruction = other_instructions
        
    def run_value_rating(self, trials, labeled_stims=None):
        instruction = visual.TextStim(self.win, 
                                      "Indicate how much you would bid in RMB",
                                      pos=[0,.7], units='norm', 
                                      height=.06)
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
                instruction.draw()
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
        self.stim_size = self.getSquareSize(self.win)
        self.presentInstruction('Welcome! Press 5 to continue...')
        
        # instructions
        
        
        # instructions for bid
        self.presentInstruction(
            """
            Finished with the first part. In this second part you will
            provide a value for each stimulus. Importantly,
            the value of the stimuli were set before you started 
            the experiment and were not based on their appearance.
            
            Each stimulus is worth between 0 RMB and 10 RMB. 
            
            Q: How will you provide your value for each stimuli?
            
            A: We will ask you to bid on the different stimuli,
                which we will explain on the next screen.
            
            When you are ready, press 5 to continue...
            """)
        
        
        self.presentInstruction(
            """
            You will start with 10 RMB, which you can use to bid. 
            
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
            combination of your original 10 RMB and the 
            results of the bid. Thus you can earn 0-20 RMB.
            
            Press 5 to continue...
            """, size=.06)
        
        # practice bdm
        self.run_bdm_explanation()
        
        self.presentInstruction(
            """
            We will now start the real bidding on the stimuli.
            
            To help you, we will tell you the value of 4 
            of the stimuli on the next screen.
            
            Press 5 to continue...
            """, size=.06)
        
        # labeling phase
        label_instruction = visual.TextStim(self.win, 
            """
            Above are the values of 4 stimuli. They
            will be shown on the screen during each of
            your bids.
                                
            When you are ready to begin press 5...
            """, pos=[0,-.3], units='norm', height=.07)
        
        labeled_stims = [(self.stim_files[i],round(self.node_values[i],1)) 
                         for i in self.labeled_nodes]
        labeled_banner = self.get_labeled_banner(labeled_stims,
                                                 [-.6,-.2, .2, .6], .4)
        for c in labeled_banner:
            c.draw()
        label_instruction.draw()
        self.win.flip()
        self.waitForKeypress()
        
        # value rating phase
        unknown_stims = []
        rating_stims = [self.stim_files[i] for i in list(set(self.node_values.keys())-set(self.labeled_nodes))]
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
        
        # determine payouts
        self.total_win = self.run_bdm(self.valuedata)
        
        # clean up and save
        taskdata = {'valuedata': self.valuedata,
                    'nodevalues': self.node_values,
                    'total_win': self.total_win,
                    'labeled_nodes': self.labeled_nodes}
        self.writeData(taskdata)
        self.presentTextToWindow('Thank you. Please wait for the experimenter',
                                 size=.1)
        self.waitForKeypress(self.quit_key)
        self.closeWindow()
        return self.total_win

