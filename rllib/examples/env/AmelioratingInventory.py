from copy import deepcopy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import ray
import time
import scipy.stats as st
import scipy.integrate as sigr
import matplotlib.pyplot as plt
import gurobipy as gb
from gurobipy import GRB
from ray.rllib.env.env_context import EnvContext
from stochastic.processes.diffusion import vasicek as vsk

#-----------------------------------#
class AmelioratingInventoryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, config=EnvContext):
        
        self.render_mode = config["render_mode"] if "render_mode" in config else None
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        #----------------------------------------------------------------------------------------------------------------------------------------------------------#
        #READ_IN PROBLEM CONFIGURATION FROM DICTIONARY
        
        #problem size
        self.numAges = config["numAges"] if "numAges" in config else 5   #the number of age classes in the inventory system
        self.nProducts = config["nProducts"] if "nProducts" in config else 2   #the number of products
        self.targetAges = config["targetAges"] if "targetAges" in config else [1,3]   #the minimum storage time of the considered products
        self.maxInventory = config["maxInventory"] if "maxInventory" in config else 50   #the maximum inventory in each age class
        
        #uncertainty distributions
        self.demandDistributions = config["demandDistributions"] if "demandDistributions" in config else [st.poisson(6.0), st.poisson(6.0)]
        self.decayProbabilities = config["decayProbabilities"] if "decayProbabilities" in config else [st.beta(a=0.7, b=34.3) for _ in range(self.numAges)]
        if isinstance(self.decayProbabilities[0], float):
            self.decayProbabilities = [st.beta(a=0.7, b=0.7*(1.0/self.decayProbabilities[i])-0.7) for i in range(self.numAges)]
            
        self.priceProcess = config["priceProcess"] if "priceProcess" in config else None
        self.priceDistribution = config["priceDistribution"] if "priceDistribution" in config else st.truncnorm(loc=250, scale=30, a=(175-250)/30, b=(325-250)/30)
        if self.priceProcess:
            processDistribution = st.norm(loc=self.priceProcess.mean(0), scale=np.sqrt((self.priceProcess.vol(0)**2)/(2*self.priceProcess.speed(0))))
            self.priceDistribution = st.truncnorm(loc=processDistribution.mean(), scale=processDistribution.std(), a=(processDistribution.ppf(0.00001)-processDistribution.mean())/processDistribution.std(), b=(processDistribution.ppf(0.99999)-processDistribution.mean())/processDistribution.std())
        
        #reward function parameters
        self.brandContributions = config["brandContributions"] if "brandContributions" in config else [300.0, 500.0]
        self.holdingCosts = config["holdingCosts"] if "holdingCosts" in config else 25
        self.outdatingCosts = config["outdatingCosts"] if "outdatingCosts" in config else 0
        self.decaySalvage = config["decaySalvage"] if "decaySalvage" in config else [50.0+i*(10.0) for i in range(self.numAges)] 
        self.salvage = config["salvage"] if "salvage" in config else [0.5*self.brandContributions[i] for i in range(self.nProducts)] 
        
        #further problem specifications
        self.allowOutdating = config["allowOutdating"] if "allowOutdating" in config else True
        self.allowBlending = config["allowBlending"] if "allowBlending" in config else False
        self.blendingRange = config["blendingRange"] if "blendingRange" in config else None
        self.allowOverage = config["allowOverage"] if "allowOverage" in config else True
        self.evaporation = config["evaporation"] if "evaporation" in config else 0.03
        
        #maximum production amount
        self.max_ppf_ff = config["max_ppf_ff"] if "max_ppf_ff" in config else 1.0
        
        #time horizon for each RL episode - initialize at 0
        self.max_horizon = config["horizon"] if "horizon" in config else 200
        self.n_steps = 0
        
        #discount factor - not needed for APO
        self.gamma = config["gamma"] if "gamma" in config else 1.0
            
        #set initial price to mean of price distribution
        self.price = self.priceDistribution.mean()
        self.meanPrice = self.priceDistribution.mean()

        #action space format (should be continuous) 
        self.action_space_design = config["action_space_design"] if "action_space_design" in config else "box_continuous"
        
        #select actions according to heuristic (circumvents RL agent)
        self.simulate_heuristic = config["simulate_heuristic"] if "simulate_heuristic" in config else False
        
        #reward shaping parameters
        self.penalty_heuristic_deviation = config["penalty_heuristic_deviation"] if "penalty_heuristic_deviation" in config else 50
        self.previous_deviation = 0
        self.penalty_structure = config["penalty_structure"] if "penalty_structure" in config else 0.5
        self.history_length = config["history_length"] if "history_length" in config else 5
        self.use_adversarial_sampling = config["use_adversarial_sampling"] if "use_adversarial_sampling" in config else True
        if self.use_adversarial_sampling:
            self.adversarial_buffer = None
        self.adversarial_threshold = int(np.floor(self.max_horizon/2 + 0.93))
        self.use_cdfs_for_regularization = False
        self.cdf_buffer = None
       
        #check environment configuration for inconsistencies
        assert len(self.targetAges) == self.nProducts
        assert len(self.demandDistributions) == self.nProducts
        assert len(self.brandContributions) == self.nProducts
        assert len(self.decayProbabilities) == self.numAges
        assert len(self.decaySalvage) == self.numAges
        assert len(self.salvage) == self.nProducts
        assert not any(i > self.numAges for i in self.targetAges)

        #possible production levels for issuance heuristic
        self.use_issuance_model = config["use_issuance_model"] if "use_issuance_model" in config else False
        self.production_step_size = config["production_step_size"] if "production_step_size" in config else 0.05
        self.time_horizon_redux = config["time_horizon_redux"] if "time_horizon_redux" in config else 0
        self.sales_bound = [self.demandDistributions[i].ppf(self.max_ppf_ff) for i in range(self.nProducts)]
        self.production_levels = {p: [round(i,2) for i in np.arange(0,self.sales_bound[p]+self.production_step_size,self.production_step_size)] for p in range(self.nProducts)}

        #determine production decisions using APO
        self.drl_for_production = config["drl_for_production"] if "drl_for_production" in config else False
        self.products_using_drl = None if not self.drl_for_production else config["products_using_drl"] if "products_using_drl" in config else [p for p in range(self.nProducts)]

        #----------------------------------------------------------------------------------------------------------------------------------------------------------#
        
        #derive function to integrate over for calculating expected revenues
        self.ff_function = {p: lambda x,p=p: x * self.demandDistributions[p].pdf(x) for p in range(self.nProducts)}
        
    	#accumulated evaporation losses
        self.evaporation_remains_per_age_class = [pow((1-self.evaporation),(i+1)) for i in range(self.numAges)]
        
        #mean decay per age class
        self.meanDecay = np.array([self.decayProbabilities[i].mean() for i in range(self.numAges)])
        
        #accumulated decay probabilities, holding costs and decay salvages over the age classes (use in lookahead LP for issuance and production)
        self.decay_remains_per_age_class = [np.multiply.reduce([(1-self.meanDecay[j]) for j in range(i+1)]) for i in range(self.numAges)]
        self.holding_acc_til_target = [sum(self.holdingCosts/np.multiply.reduce([(1-self.meanDecay[self.targetAges[p]-j]) for j in range(i+1)]) for i in range(self.targetAges[p]+1)) for p in range(self.nProducts)]
        self.holding_acc_til_age = [sum(self.holdingCosts/np.multiply.reduce([(1-self.meanDecay[i-k]) for k in range(j+1)]) for j in range(i+1)) for i in range(self.numAges)]
        self.decaySalvage_acc_til_target = [sum(self.decaySalvage[self.targetAges[p]-i]*self.meanDecay[self.targetAges[p]-i]/np.multiply.reduce([(1-self.meanDecay[self.targetAges[p]-j]) for j in range(i+1)]) for i in range(self.targetAges[p]+1)) for p in range(self.nProducts)] 

        #expected revenues (piecewise-linear approximation for lookahead LPs)
        self.expected_revenue_salvage = {p: {l: self.brandContributions[p] * (sigr.quad(self.ff_function[p], 0, l)[0] + l * (1-self.demandDistributions[p].cdf(l))) + self.salvage[p] * (l*self.demandDistributions[p].cdf(l) - sigr.quad(self.ff_function[p], 0, l)[0]) for l in self.production_levels[p]} for p in range(self.nProducts)} 

        #parameters for scaling rewards; #initial inventory = mean demands
        self.max_reward, self.inventory_position = upper_bound(self)
        self.min_reward = 0
        self.reward_lb = config["reward_lb"] if "reward_lb" in config else -10.0
        self.reward_ub = config["reward_ub"] if "reward_ub" in config else 10.0
        print("MAX REWARD: ", self.max_reward)
        print("MIN_REWARD: ", self.min_reward)
        
        #set lookahead horizon
        self.time_steps = self.numAges + 1 - self.time_horizon_redux
        #initialize heuristic LP
        self.create_heuristic_lp()

        #derive heuristic action from lookahead model and use heuristic action to initial artificial action "history"
        self.heuristic_model.optimize()
        if self.use_issuance_model:
            if not self.drl_for_production:
                self.heuristic_action = [self.heuristic_model.getVarByName("inv[0,1]").X/(1-self.meanDecay[0])]
                self.action_history = np.array([[self.heuristic_action[0]/self.maxInventory] for i in range(self.history_length)])
            else:
                self.heuristic_action = [self.heuristic_model.getVarByName("inv[0,1]").X/(1-self.meanDecay[0])] + [sum(self.heuristic_model.getVarByName(f"iss[{p},{a},0]").X for a in range(self.numAges)) for p in range(self.nProducts)]
                self.action_history = np.array([[self.heuristic_action[0]/self.maxInventory]+[self.heuristic_action[p+1]/self.sales_bound[p] for p in range(self.nProducts)] for i in range(self.history_length)])
        else:
            self.heuristic_action = [self.heuristic_model.getVarByName("inv[0,1]").X/(1-self.meanDecay[0])] + [sum(self.heuristic_model.getVarByName(f"iss[{p},{a},0]").X for p in range(self.nProducts)) for a in range(self.numAges)]
            self.action_history = np.array([[self.heuristic_action[0]/self.maxInventory]+[sum(self.heuristic_model.getVarByName(f"iss[{p},{a},0]").X for a in range(self.numAges))/self.sales_bound[p] for p in range(self.nProducts)] for i in range(self.history_length)])

        print("starting inventory: ", self.inventory_position)
        
        #create initial price and inventory "history"
        self.just_purchased = self.heuristic_action[0]
        self.price_history = np.array([self.priceDistribution.cdf(self.priceDistribution.mean()) for i in range(self.history_length)])
        self.inventory_history = np.array([self.inventory_position for i in range(self.history_length)])
        self.starting_state = self._get_obs()

        #create action space and state space
        if self.action_space_design == "box_continuous":
            if self.use_issuance_model:
                #initialize lookahead LP for issuance/production
                if not self.drl_for_production:
                    #if the lookahead LP is used for issuance and production actions, the action space only considers the purchasing action
                    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,)) 
                else:
                    #if the lookahead LP is used for issuance actions, the action space considers purchasing and production actions
                    self.just_produced = self.heuristic_action[1:]
                    self.action_space = spaces.Box(low=np.full((len(self.products_using_drl)+1,),-1.0), high=np.full((len(self.products_using_drl)+1,),1.0), shape=(len(self.products_using_drl)+1,)) 
                self.create_issuance_lp()
            else:
                #otherwise issuance volumes from all applicable age classes are included in the action space
                self.create_allocation_lp()
                if self.allowBlending:
                    self.action_space = spaces.Box(low=np.full((self.numAges+1,),-1.0), high=np.full((self.numAges+1,),1.0), shape=(self.numAges+1,))
                else:
                    self.action_space = spaces.Box(low=np.full((self.numAges-self.targetAges[0]+1,),-1.0), high=np.full((self.numAges-self.targetAges[0]+1,),1.0), shape=(self.numAges-self.targetAges[0]+1,))
            
            #when using history-based structural reward shaping, include the history in the state space
            if self.penalty_structure > 0:
                original_space = spaces.Dict({
                    "price": spaces.Box(low=0.0, high=1.0, shape = (1,)),
                    "inventory": spaces.Box(low=np.full((self.numAges,),0.0), high=np.full((self.numAges,),1.0), shape=(self.numAges,)),
                    "action_history": spaces.Box(low=np.full((self.history_length,1),0.0), high=np.full((self.history_length,1),1.0), shape=(self.history_length,1)) if (self.use_issuance_model and not self.drl_for_production)
                                        #else spaces.Box(low=np.full((self.history_length, self.numAges+1),0.0), high=np.full((self.history_length, self.numAges+1),1.0), shape=(self.history_length,self.numAges+1)) if self.allowBlending
                                        else spaces.Box(low=np.full((self.history_length, self.nProducts+1),0.0), high=np.full((self.history_length, self.nProducts+1),1.0), shape=(self.history_length,self.nProducts+1)),
                    "price_history": spaces.Box(low=np.full((self.history_length,),0.0),high=np.full((self.history_length,),1.0),shape=(self.history_length,)),
                    "inventory_history": spaces.Box(low=np.full((self.history_length,self.numAges),0.0), high=np.full((self.history_length,self.numAges),1.0), shape=(self.history_length,self.numAges)),
                })
            else:
                original_space = spaces.Dict({
                    "price": spaces.Box(low=0.0, high=1.0, shape = (1,)),
                    "inventory": spaces.Box(low=np.full((self.numAges,),0.0), high=np.full((self.numAges,),1.0), shape=(self.numAges,)),
                })
            
            self.observation_space = original_space
        else:
            raise ValueError("Not Implemented!")

    def get_heuristic_action(self):
        #update the heuristic model to new inventory and price levels
        self.update_heuristic_model()
        #solve heuristic model
        self.heuristic_model.optimize()
        #get purchasing volume 
        purchasing = self.heuristic_model.getVarByName("inv[0,1]").X/(1-self.meanDecay[0])
        #get issuance decisions
        issuance_product = {p:[self.heuristic_model.getVarByName(f"iss[{p},{a},0]").X for a in range(self.numAges)] for p in range(self.nProducts)}
        return purchasing, issuance_product
    
    #derive violations of desired policy structure in history
    def get_structure_deviation(self, purchasing, production):
        curr_p = self.priceDistribution.cdf(self.price)
        purchasing_deviation = 0
        production_deviation = np.array([0 for i in range(self.nProducts)])
        for i in range(self.history_length):
            hist_a = self.action_history[i][0] * self.maxInventory
            hist_f = [self.action_history[i][p] * self.sales_bound[p] for p in range(self.nProducts)]
            hist_p = self.price_history[i]
            hist_inv = self.inventory_history[i]
            if (purchasing > hist_a and curr_p >= hist_p and not any(self.inventory_position[a] < hist_inv[a] for a in range(self.numAges))) or (purchasing < hist_a and curr_p <= hist_p and not any(self.inventory_position[a] > hist_inv[a] for a in range(self.numAges))):
                purchasing_deviation += abs(purchasing-hist_a)
            for p in range(self.nProducts):
                if (production[p] > hist_f[p] and not any(self.inventory_position[a] > hist_inv[a] for a in range(self.numAges))) or (production[p] < hist_f[p] and not any(self.inventory_position[a] < hist_inv[a] for a in range(self.numAges))):
                    production_deviation[p] += abs(production[p]-hist_f[p]) 
            
        return purchasing_deviation + sum(production_deviation)

    #derive violations of desired policy structure in history (only purchasing)
    def get_structure_deviation_purchasing(self, purchasing):
        curr_p = self.priceDistribution.cdf(self.price)
        purchasing_deviation = 0
        for i in range(self.history_length):
            hist_a = self.action_history[i][0] * self.maxInventory
            hist_p = self.price_history[i]
            hist_inv = self.inventory_history[i]

            if (purchasing > hist_a and curr_p >= hist_p and not any(self.inventory_position[a] < hist_inv[a] for a in range(self.numAges))) or (purchasing < hist_a and curr_p <= hist_p and not any(self.inventory_position[a] > hist_inv[a] for a in range(self.numAges))):
                purchasing_deviation += abs(purchasing-hist_a)

        return purchasing_deviation

    #update action, price, and inventory history
    def update_history(self,purchasing,production):
        self.action_history = np.vstack(([purchasing/self.maxInventory]+[production[p]/self.sales_bound[p] for p in range(self.nProducts)],self.action_history[:-1]))
        self.price_history = np.insert(self.price_history[:-1],0,self.priceDistribution.cdf(self.price))
        self.inventory_history = np.vstack((self.inventory_position,self.inventory_history[:-1]))

    #update action (only purchasing), price, and inventory history
    def update_history_purchasing(self,purchasing):
        self.action_history = np.vstack(([purchasing/self.maxInventory],self.action_history[:-1]))
        self.price_history = np.insert(self.price_history[:-1],0,self.priceDistribution.cdf(self.price))
        self.inventory_history = np.vstack((self.inventory_position,self.inventory_history[:-1]))
    
    #internal getter for state
    def  _get_obs(self):
        if self.penalty_structure > 0:
            return {"price": np.array([self.priceDistribution.cdf(self.price)]), "inventory": self.inventory_position/self.maxInventory, "action_history":self.action_history, "price_history":self.price_history, "inventory_history":self.inventory_history/self.maxInventory}
        else:
            return {"price": np.array([self.priceDistribution.cdf(self.price)]), "inventory": self.inventory_position/self.maxInventory} 

    def _get_state_from_obs(self, obs):
        self.inventory_position = obs["inventory"] * self.maxInventory
        self.price = self.priceDistribution.ppf(obs["price"][0])
        if self.penalty_structure > 0:
            self.action_history = obs["action_history"]
            self.price_history = obs["price_history"]
            self.inventory_history = obs["inventory_history"]*self.maxInventory

    #reset function does nothing --> infinite horizon
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.use_adversarial_sampling and self.n_steps == self.adversarial_threshold:
            print("UPDATE STATE IN RESET FUNC")
            self._get_state_from_obs(self.starting_state)
            self.price = self.priceDistribution.ppf(1-self.priceDistribution.cdf(self.price))
        else:
            print("FINISH EPISODE")
            self.adversarial_buffer = None
            self.starting_state = self._get_obs()
            self.n_steps = 0
        return self._get_obs(), {}

    #step function: if specified, use lookahead LP to approximate issuance decisions
    def step(self, action):
        if self.action_space_design == "box_continuous":
            if self.use_issuance_model:
                return self.step_continuous_issuance_lp(action)
            else:
                return self.step_continuous(action)
        else:
            return None

    def step_continuous(self, action):
        self.n_steps += 1
        if self.n_steps > self.max_horizon:
            self.n_steps = 1
        if self.use_adversarial_sampling and self.n_steps == 1:
            self.adversarial_buffer = None
            self.starting_state = self._get_obs()

        #get action from heuristic
        if self.simulate_heuristic:
            action = self.get_heuristic_action()
            purchasing = deepcopy(action[0])
            issuance_product = deepcopy(action[1])
        #map neural network output to implementable action
        else:
            purchasing = ((action[0]+1)/2) * (self.maxInventory)
            #mask issuance actions using current inventory levels
            issuance = [((action[1+i]+1)/2) * self.inventory_position[i+self.targetAges[0]] for i in range(self.numAges-self.targetAges[0])]
            if self.allowBlending:
                issuance = self.get_issuance_blending(action[1:], issuance)
            else:
                issuance = [0 for _ in range(self.targetAges[0])] + issuance
        
            #use LP model to allocate issuance volumes to products
            issuance_product = self.allocate_issuance(issuance) 
             
        #derive production/production volumes and post-decision inventory
        production = [max(0,min(self.sales_bound[p],sum(issuance_product[p][a] * self.evaporation_remains_per_age_class[a] for a in range(self.numAges)))) for p in range(self.nProducts)]
        new_inventory = np.nan_to_num(np.array([max(0,self.inventory_position[a] - sum(issuance_product[p][a] for p in range(self.nProducts))) for a in range(self.numAges)]))
        outdating = new_inventory[-1]
        
        #sample demand
        demand = [float(self.demandDistributions[i].rvs()) for i in range(self.nProducts)]
        sales = [min(production[i],demand[i]) for i in range(self.nProducts)]
        leftovers = [max(0,demand[i] - production[i]) for i in range(self.nProducts)]
        
        #quantify violations of desired policy structure in history
        if self.penalty_structure > 0:
            structure_deviation = self.get_structure_deviation(purchasing, production)
            self.update_history(purchasing, production)
        else:
            structure_deviation = 0
        
        #quantify deviation of current action from heuristic action
        if not self.simulate_heuristic and self.penalty_heuristic_deviation > 0: 
            heuristic_action = self.get_heuristic_action()
            heuristic_deviation = -self.penalty_heuristic_deviation*(abs(purchasing - heuristic_action[0]) + sum(abs(sum(issuance_product[p][i]-heuristic_action[1][p][i] for p in range(self.nProducts))) for i in range(self.numAges)))
        else:
            heuristic_action = None
            heuristic_deviation = 0
        
        #inventories age by one period
        pre_decay_inventory = np.nan_to_num(np.concatenate(([purchasing],new_inventory[:-1])))
        
        #sample decay
        if self.use_cdfs_for_regularization:
            decay_proportions = np.array([self.decayProbabilities[i].ppf(self.cdf_buffer[self.n_steps-1][i+1]) for i in range(self.numAges)])
        elif self.use_adversarial_sampling and self.n_steps > self.adversarial_threshold:
            decay_proportions = np.array([self.decayProbabilities[i].ppf(self.adversarial_buffer[self.n_steps-self.adversarial_threshold - 1][i+1]) for i in range(self.numAges)])
        else:
            decay_proportions = np.array([self.decayProbabilities[i].rvs() for i in range(self.numAges)])
        decay_samples = pre_decay_inventory * decay_proportions
            
        #subtract decay from inventory
        self.inventory_position = (pre_decay_inventory - decay_samples)
        
        #compute and normalize reward
        purchasing_cost = purchasing*self.price
        revenue = sum(sigr.quad(self.ff_function[p], 0, production[p])[0]*(self.brandContributions[p] - self.salvage[p]) + production[p] * self.brandContributions[p] * (1-self.demandDistributions[p].cdf(production[p])) + self.salvage[p] * production[p] * self.demandDistributions[p].cdf(production[p])  for p in range(self.nProducts)) 
        holding_cost = sum(pre_decay_inventory)*self.holdingCosts
        decay_salvage = np.dot(decay_samples, (self.decaySalvage))
        outdating_cost = outdating*self.outdatingCosts 

        reward = revenue - purchasing_cost - holding_cost + decay_salvage - outdating_cost + (heuristic_deviation - (self.previous_deviation/self.gamma)) - self.penalty_structure*structure_deviation
        norm_reward = (reward - self.min_reward)/(self.max_reward-self.min_reward) * (self.reward_ub-self.reward_lb) + self.reward_lb  

        #update heuristic deviation
        self.previous_deviation = deepcopy(heuristic_deviation)
        #sample price
        prev_price = deepcopy(self.price)
        if self.use_cdfs_for_regularization:
            self.price = self.priceDistribution.ppf(self.cdf_buffer[self.n_steps-1][0])
        elif self.use_adversarial_sampling and self.n_steps > self.adversarial_threshold:
            self.price = self.priceDistribution.ppf(self.adversarial_buffer[self.n_steps - self.adversarial_threshold-1][0])
        else:
            if self.priceProcess:
                self.price = self.priceProcess.sample(1, initial=self.price)[1]
            else:
                self.price = self.priceDistribution.rvs()

        #adversarial sampling - in first half of trajectory, add adversarial samples to buffer which can be simply selected in second half
        cdf_vector = np.concatenate(([self.priceDistribution.cdf(self.price)],[self.decayProbabilities[i].cdf(decay_proportions[i]) for i in range(self.numAges)]))
        if self.use_adversarial_sampling and self.n_steps <= self.adversarial_threshold:
            if self.n_steps == 1:
                self.adversarial_buffer = 1-cdf_vector
            else:
                self.adversarial_buffer = np.vstack((self.adversarial_buffer, 1-cdf_vector))
                
        #update state
        observation = self._get_obs()

        return observation, norm_reward, False, self.n_steps == self.max_horizon or (self.use_adversarial_sampling and self.n_steps == self.adversarial_threshold), {"revenue":revenue, "purchasing_cost": purchasing_cost, "holding_cost":holding_cost, "decay_salvage":decay_salvage, "outdating_cost": outdating_cost, "purchasing":purchasing, "production":production, "issuance":[sum(issuance_product[p][a] for p in range(self.nProducts)) for a in range(self.numAges)], "demand": demand, "outdating": outdating, "overproduction": leftovers, "inventory": self.inventory_position, "sales": sales, "price":prev_price, "decay_proportions":decay_proportions, "cdf_vector":cdf_vector}

    def step_continuous_issuance_lp(self, action):
        
        # start_time = time.time()

        self.n_steps += 1
        if self.n_steps > self.max_horizon:
            self.n_steps = 1
        if self.n_steps == 1:
            self.adversarial_buffer = None
            self.starting_state = self._get_obs()
            #print("STARTING STATE ENV: ", self.price, " ", self.inventory_position)
            
        production = None
        starting_inventory = self.inventory_position
        #get purchasing action from heuristic
        if self.simulate_heuristic:
            action = self.get_heuristic_action()
            purchasing = deepcopy(action[0])
            if self.drl_for_production:
                production = [sum(action[1][self.products_using_drl[p]][a] * self.evaporation_remains_per_age_class[a] for a in range(self.numAges))for p in range(len(self.products_using_drl))]
         
        #derive purchasing (and possibly production) volumes from neural network output  
        else:
            purchasing = ((action[0]+1)/2) * self.maxInventory
            purchasing = np.nan_to_num(purchasing)
            if self.drl_for_production:
                production = [(action[p+1]+1)/2 *  self.sales_bound[self.products_using_drl[p]] for p in range(len(self.products_using_drl))]
            
        self.just_purchased = purchasing
        self.just_produced = production
        
        # print("TIME TO READ ACTION: ", time.time() - start_time)
        # start_time = time.time()

        #update issuance LP to new inventory level and purchasing level
        self.update_issuance_model()

        # print("TIME TO UPDATE ISSUANCE MODEL: ", time.time() - start_time)
        # start_time = time.time()

        #optmize model
        self.issuance_model.optimize()

        # print("TIME TO SOLVE ISSUANCE MODEL: ", time.time() - start_time)
        # start_time = time.time()
        
        #get issuance decisions
        issuance_product = {p: {a: self.issuance_model.getVarByName(f"iss[{p},{a},0]").X for a in range(self.numAges)} for p in range(self.nProducts)}
        
        #derive production/production volumes and post-decision inventory    
        production = [min(self.sales_bound[p],sum(issuance_product[p][a] * self.evaporation_remains_per_age_class[a] for a in range(self.numAges))) for p in range(self.nProducts)]
        new_inventory = np.nan_to_num(np.array([max(0,self.inventory_position[a] - sum(issuance_product[p][a] for p in range(self.nProducts))) for a in range(self.numAges)]))
        outdating = new_inventory[-1]

        #sample demand
        demand = [float(self.demandDistributions[i].rvs()) for i in range(self.nProducts)]
        sales = [min(production[i],demand[i]) for i in range(self.nProducts)]
        leftovers = [max(0,demand[i] - production[i]) for i in range(self.nProducts)]

        #quantify violations of desired policy structure in history
        if self.penalty_structure > 0:
            if self.drl_for_production:
                structure_deviation = self.get_structure_deviation(purchasing, production)
            else:
                structure_deviation = self.get_structure_deviation_purchasing(purchasing)
        else:
            structure_deviation = 0

        if self.drl_for_production:
            self.update_history(purchasing, production)
        else:
            self.update_history_purchasing(purchasing)

        #quantify deviation of current action from heuristic action
        if not self.simulate_heuristic and self.penalty_heuristic_deviation > 0: 
            heuristic_action = self.get_heuristic_action()[0]
            heuristic_deviation = -self.penalty_heuristic_deviation*abs(purchasing-heuristic_action)
        else:
            heuristic_action = None
            heuristic_deviation = 0
        
        #sample decay
        pre_decay_inventory = np.nan_to_num(np.concatenate(([purchasing],new_inventory[:-1])))
        if self.use_cdfs_for_regularization:
            decay_proportions = np.array([self.decayProbabilities[i].ppf(self.cdf_buffer[self.n_steps-1][i+1]) for i in range(self.numAges)])
        elif self.use_adversarial_sampling and self.n_steps > self.adversarial_threshold:
            decay_proportions = np.array([self.decayProbabilities[i].ppf(self.adversarial_buffer[self.n_steps-self.adversarial_threshold - 1][i+1]) for i in range(self.numAges)])
        else:
            decay_proportions = np.array([self.decayProbabilities[i].rvs() for i in range(self.numAges)])
        decay_samples = pre_decay_inventory * decay_proportions

        #subtract decay from inventory    
        self.inventory_position = (pre_decay_inventory - decay_samples)

        #compute and normalize rewards
        purchasing_cost = purchasing*self.price
        revenue = sum(sigr.quad(self.ff_function[p], 0, production[p])[0]*(self.brandContributions[p] - self.salvage[p]) + production[p] * self.brandContributions[p] * (1-self.demandDistributions[p].cdf(production[p])) + self.salvage[p] * production[p] * self.demandDistributions[p].cdf(production[p])  for p in range(self.nProducts)) 
        holding_cost = sum(pre_decay_inventory)*self.holdingCosts
        decay_salvage = np.dot(decay_samples, (self.decaySalvage))
        outdating_cost = outdating*self.outdatingCosts 

        reward = revenue - purchasing_cost - holding_cost + decay_salvage - outdating_cost + (heuristic_deviation - (self.previous_deviation/self.gamma)) - self.penalty_structure*structure_deviation
        norm_reward = (reward - self.min_reward)/(self.max_reward-self.min_reward) * (self.reward_ub-self.reward_lb) + self.reward_lb  

        #update heuristic deviation
        self.previous_deviation = deepcopy(heuristic_deviation)
        
        #sample price
        prev_price = deepcopy(self.price)
        if self.use_cdfs_for_regularization:
            self.price = self.priceDistribution.ppf(self.cdf_buffer[self.n_steps-1][0])
        elif self.use_adversarial_sampling and self.n_steps > self.adversarial_threshold:
            self.price = self.priceDistribution.ppf(self.adversarial_buffer[self.n_steps - self.adversarial_threshold-1][0])
        else:
            if self.priceProcess:
                self.price = self.priceProcess.sample(1, initial=self.price)[1]
            else:
                self.price = self.priceDistribution.rvs()
                
        #adversarial sampling - in first half of trajectory, add adversarial samples to buffer which can be simply selected in second half
        cdf_vector = np.concatenate(([self.priceDistribution.cdf(self.price)],[self.decayProbabilities[i].cdf(decay_proportions[i]) for i in range(self.numAges)]))
        if self.use_adversarial_sampling and self.n_steps <= self.adversarial_threshold:
            if self.n_steps == 1:
                self.adversarial_buffer = 1-cdf_vector
            else:
                self.adversarial_buffer = np.vstack((self.adversarial_buffer, 1-cdf_vector))
                    
        #update state        
        observation = self._get_obs()

        # print("TIME FOR REWARD CALCULATIONS AND TRANSITION SAMPLES: ", time.time() - start_time)
        
        return observation, norm_reward, False, self.n_steps == self.max_horizon or (self.use_adversarial_sampling and self.n_steps == self.adversarial_threshold), {"revenue":revenue, "purchasing_cost": purchasing_cost, "holding_cost":holding_cost, "decay_salvage":decay_salvage, "outdating_cost": outdating_cost, "purchasing":purchasing, "production":production, "issuance":[sum(issuance_product[p][a] for a in range(self.numAges)) for p in range(self.nProducts)], "demand": demand, "outdating": outdating, "overproduction": leftovers, "inventory": starting_inventory, "price":prev_price, "sales": sales, "decay_proportions":decay_proportions, "cdf_vector":cdf_vector}

    #update issuance LP to new inventory and purchasing input
    def update_issuance_model(self):
        self.issuance_model.update()
        self.update_inventory_constraints(self.issuance_model)
        self.issuance_model.remove(self.issuance_model.getConstrByName("startp"))
        inv = self.issuance_model.getVarByName("inv[0,1]")
        self.issuance_model.addLConstr(inv == self.just_purchased * (1-self.meanDecay[0]), name="startp")
        if self.drl_for_production:
            for p in range(len(self.products_using_drl)):
                prod = self.products_using_drl[p]
                self.issuance_model.remove(self.issuance_model.getConstrByName("production"+str(prod)))
                prod_vio = self.issuance_model.getVarByName(f"prod_vio[{prod}]")
                iss = [self.issuance_model.getVarByName(f"iss[{prod},{a},0]") for a in range(self.numAges)]
                self.issuance_model.addLConstr(gb.quicksum(iss[a] * self.evaporation_remains_per_age_class[a] for a in range(self.numAges)) == self.just_produced[prod] - prod_vio, name="production"+str(prod))
        self.issuance_model.update()

    #update heuristic LP to new inventory and price
    def update_heuristic_model(self):
        self.heuristic_model.update()
        self.update_inventory_constraints(self.heuristic_model)
        for p in range(self.nProducts):
            for t in range(self.time_steps-1):
                purchase = self.heuristic_model.getVarByName(f"iss[{p},{t},{t+1}]")
                purchase.Obj = ((-self.price)/(np.multiply.reduce([1-self.meanDecay[t-j] for j in range(t+1)]))) + sum((self.decaySalvage[t-j]*self.meanDecay[t-j]-self.holdingCosts)/(np.multiply.reduce([1-self.meanDecay[t-k] for k in range(j+1)])) for j in range(t+1))
        self.heuristic_model.update()

    #update LP constraints to new inventory    
    def update_inventory_constraints(self, model):  
        for a in range(self.numAges):
            model.remove(model.getConstrByName("start"+str(a)))
            inv = model.getVarByName(f"inv[{a},0]")
            model.addLConstr(inv == self.inventory_position[a], name="start"+str(a))

    #create a lookahead linear program for taking issuance/production decisions given the current state and purchasing action as inputs
    def create_issuance_lp(self, time_horizon_redux = 0):
        self.issuance_model = gb.Model()
        #disable printout of Gurobi solution process
        self.issuance_model.setParam('OutputFlag', 1)
        
        #prepare indices of decision variables
        tuplelist_iss = []
        tuplelist_ff = []
        for p in range(self.nProducts):
            for a in range(self.numAges):
                for t in range(self.time_steps):
                    tuplelist_iss.append((p,a,t))
            for l in self.production_levels[p]:
                for t in range(self.time_steps):
                    tuplelist_ff.append((p,l,t))

        #define decision variables
        inv = self.issuance_model.addVars(self.numAges, self.time_steps, name="inv")
        iss = self.issuance_model.addVars(tuplelist_iss, name="iss")
        out = self.issuance_model.addVars(self.time_steps, name="out")
        ff = self.issuance_model.addVars(tuplelist_ff, lb=[0.0 for i in range(len(tuplelist_ff))], ub=[1.0 for i in range(len(tuplelist_ff))], name="ff")
        if self.drl_for_production:
            prod_violation = self.issuance_model.addVars(self.products_using_drl, lb=[0.0 for _ in self.products_using_drl], ub=[self.sales_bound[p] for p in self.products_using_drl], name="prod_vio")

        print(self.time_steps)
        print(self.numAges)
        print(self.nProducts)
        print(len(tuplelist_ff))
        print(len(tuplelist_iss))

        #starting inventory
        for a in range(self.numAges):
            self.issuance_model.addLConstr(inv[a,0] == self.inventory_position[a], name="start"+str(a))

        #purchasing volume from actor network
        self.issuance_model.addLConstr(inv[0,1] == self.just_purchased * (1-self.meanDecay[0]), name="startp")
        
        #maximum purchasing/inventory capacity restrictions
        for t in range(1,self.time_steps):
            self.issuance_model.addLConstr(inv[0,t] <= (self.maxInventory)*(1-self.meanDecay[0]))

        for t in range(self.time_steps):
            #outdating
            self.issuance_model.addLConstr(out[t] >= inv[self.numAges-1,t] - gb.quicksum(iss[p,self.numAges-1,t] for p in range(self.nProducts)))
            for p in range(self.nProducts):
                #blending restrictions
                if not self.allowBlending:
                    self.issuance_model.addLConstr(gb.quicksum(iss[p,a,t] for a in range(self.targetAges[p])) <= 0)
                elif self.blendingRange is not None:
                    self.issuance_model.addLConstr(gb.quicksum(iss[p,a,t] for a in [i for i in range(self.numAges) if i not in range(self.targetAges[p]-self.blendingRange, self.targetAges[p]+self.blendingRange+1)]) <= 0)
                #target age restrictions
                self.issuance_model.addLConstr(gb.quicksum(iss[p,a,t] * (a) * self.evaporation_remains_per_age_class[a] for a in range(self.numAges)) >= self.targetAges[p] * gb.quicksum(iss[p,a,t] * self.evaporation_remains_per_age_class[a] for a in range(self.numAges)))
                #target age excess restrictions
                # if not p == self.nProducts - 1:
                #     self.issuance_model.addLConstr(gb.quicksum(iss[p,a,t] for a in range(self.targetAges[p+1], self.numAges)) <= 0)
                #restrict production to quasi-binary
                self.issuance_model.addLConstr(gb.quicksum(ff[p,l,t] for l in self.production_levels[p]) <= 1.0)
                #relate production to issuance volumes
                self.issuance_model.addLConstr(gb.quicksum(ff[p,l,t] * l for l in self.production_levels[p]) <= gb.quicksum(iss[p,a,t] * self.evaporation_remains_per_age_class[a] for a in range(self.numAges)))
            for a in range(self.numAges):
                #inventory balance
                if t>0 and a>0:
                    self.issuance_model.addLConstr(inv[a,t] == (inv[a-1,t-1] - gb.quicksum(iss[p,a-1,t-1] for p in range(self.nProducts)))*(1-self.meanDecay[a]))
                #restrict issuance by inventory volumes
                self.issuance_model.addLConstr(gb.quicksum(iss[p,a,t] for p in range(self.nProducts)) <= inv[a,t])
                
        # restrict issuance by (optionally provided) production volumes
        if self.drl_for_production:
            for p in self.products_using_drl:
                self.issuance_model.addLConstr(gb.quicksum(iss[p,a,0] * self.evaporation_remains_per_age_class[a] for a in range(self.numAges)) == self.just_produced[p] - prod_violation[p], name="production"+str(p))
            obj = gb.quicksum(ff[p,l,t] * self.expected_revenue_salvage[p][l] for p in range(self.nProducts) for l in self.production_levels[p] for t in range(self.time_steps)) + gb.quicksum(iss[p,a,t]*(sum((self.decaySalvage[a-j]*self.meanDecay[a-j]-self.holdingCosts)/(np.multiply.reduce([1-self.meanDecay[a-k] for k in range(j+1)])) for j in range(a+1)) - self.meanPrice/(np.multiply.reduce([1-self.meanDecay[a-j] for j in range(a+1)]))) for p in range(self.nProducts) for a in range(self.numAges) for t in range(self.time_steps)) + gb.quicksum(out[t]*(-self.outdatingCosts+sum((self.decaySalvage[self.numAges-1-j]*self.meanDecay[self.numAges-1-j]-self.holdingCosts)/(np.multiply.reduce([1-self.meanDecay[self.numAges-1-k] for k in range(j+1)])) for j in range(self.numAges)) - self.priceDistribution.mean()/(np.multiply.reduce([1-self.meanDecay[self.numAges-1-j] for j in range(self.numAges)]))) for t in range(self.time_steps)) - gb.quicksum(prod_violation[p] * self.brandContributions[p] * self.time_steps for p in self.products_using_drl)
        else:
            obj = gb.quicksum(ff[p,l,t] * self.expected_revenue_salvage[p][l] for p in range(self.nProducts) for l in self.production_levels[p] for t in range(self.time_steps)) + gb.quicksum(iss[p,a,t]*(sum((self.decaySalvage[a-j]*self.meanDecay[a-j]-self.holdingCosts)/(np.multiply.reduce([1-self.meanDecay[a-k] for k in range(j+1)])) for j in range(a+1)) - self.meanPrice/(np.multiply.reduce([1-self.meanDecay[a-j] for j in range(a+1)]))) for p in range(self.nProducts) for a in range(self.numAges) for t in range(self.time_steps)) + gb.quicksum(out[t]*(-self.outdatingCosts+sum((self.decaySalvage[self.numAges-1-j]*self.meanDecay[self.numAges-1-j]-self.holdingCosts)/(np.multiply.reduce([1-self.meanDecay[self.numAges-1-k] for k in range(j+1)])) for j in range(self.numAges)) - self.priceDistribution.mean()/(np.multiply.reduce([1-self.meanDecay[self.numAges-1-j] for j in range(self.numAges)]))) for t in range(self.time_steps))
        
        self.issuance_model.setObjective(obj, GRB.MAXIMIZE)

    #create a lookahead linear program for taking all decisions
    def create_heuristic_lp(self, time_horizon_redux = 0):
        self.heuristic_model = gb.Model()
        #disable printout of Gurobi solution process
        self.heuristic_model.setParam('OutputFlag', 0)
        
        #prepare indices of decision variables
        tuplelist_iss = []
        tuplelist_ff = []
        for p in range(self.nProducts):
            for a in range(self.numAges):
                for t in range(self.time_steps):
                    tuplelist_iss.append((p,a,t))
            for l in self.production_levels[p]:
                for t in range(self.time_steps):
                    tuplelist_ff.append((p,l,t))

        #define decision variables
        inv = self.heuristic_model.addVars(self.numAges, self.time_steps, name="inv")
        iss = self.heuristic_model.addVars(tuplelist_iss, name="iss")
        out = self.heuristic_model.addVars(self.time_steps, name="out")
        ff = self.heuristic_model.addVars(tuplelist_ff, lb=[0.0 for i in range(len(tuplelist_ff))], ub=[1.0 for i in range(len(tuplelist_ff))], name="ff")

        #starting inventory
        for a in range(self.numAges):
            self.heuristic_model.addLConstr(inv[a,0] == self.inventory_position[a], name="start"+str(a))
        
        #maximum purchasing/inventory capacity restrictions
        for t in range(1,self.time_steps):
            self.heuristic_model.addLConstr(inv[0,t] <= (self.maxInventory)*(1-self.meanDecay[0]))

        for t in range(self.time_steps):
            #outdating
            self.heuristic_model.addLConstr(out[t] >= inv[self.numAges-1,t] - gb.quicksum(iss[p,self.numAges-1,t] for p in range(self.nProducts)))
            for p in range(self.nProducts):
                #blending restrictions
                if not self.allowBlending:
                    self.heuristic_model.addLConstr(gb.quicksum(iss[p,a,t] for a in range(self.targetAges[p])) <= 0)
                elif self.blendingRange is not None:
                    self.heuristic_model.addLConstr(gb.quicksum(iss[p,a,t] for a in [i for i in range(self.numAges) if i not in range(self.targetAges[p]-self.blendingRange, self.targetAges[p]+self.blendingRange+1)]) <= 0)
                #target age restrictions
                self.heuristic_model.addLConstr(gb.quicksum(iss[p,a,t] * (a) * self.evaporation_remains_per_age_class[a] for a in range(self.numAges)) >= self.targetAges[p] * gb.quicksum(iss[p,a,t] * self.evaporation_remains_per_age_class[a] for a in range(self.numAges)))
                #target age excess restrictions
                # if not p == self.nProducts - 1:
                #     self.heuristic_model.addLConstr(gb.quicksum(iss[p,a,t] for a in range(self.targetAges[p+1], self.numAges)) <= 0)
                #restrict production to quasi-binary
                self.heuristic_model.addLConstr(gb.quicksum(ff[p,l,t] for l in self.production_levels[p]) <= 1.0)
                #relate production to issuance volumes
                self.heuristic_model.addLConstr(gb.quicksum(ff[p,l,t] * l for l in self.production_levels[p]) <= gb.quicksum(iss[p,a,t] * self.evaporation_remains_per_age_class[a] for a in range(self.numAges)))
            for a in range(self.numAges):
                #inventory balance
                if t>0 and a>0:
                    self.heuristic_model.addLConstr(inv[a,t] == (inv[a-1,t-1] - gb.quicksum(iss[p,a-1,t-1] for p in range(self.nProducts)))*(1-self.meanDecay[a]))
                #restrict issuance by inventory level
                self.heuristic_model.addLConstr(gb.quicksum(iss[p,a,t] for p in range(self.nProducts)) <= inv[a,t])
                
        #set objective value       
        obj = gb.quicksum(ff[p,l,t] * self.expected_revenue_salvage[p][l] for p in range(self.nProducts) for l in self.production_levels[p] for t in range(self.time_steps)) + gb.quicksum(iss[p,a,t]*(sum((self.decaySalvage[a-j]*self.meanDecay[a-j]-self.holdingCosts)/(np.multiply.reduce([1-self.meanDecay[a-k] for k in range(j+1)])) for j in range(a+1)) - self.meanPrice/(np.multiply.reduce([1-self.meanDecay[a-j] for j in range(a+1)]))) for p in range(self.nProducts) for a in range(self.numAges) for t in range(self.time_steps)) + gb.quicksum(out[t]*(-self.outdatingCosts+sum((self.decaySalvage[self.numAges-1-j]*self.meanDecay[self.numAges-1-j]-self.holdingCosts)/(np.multiply.reduce([1-self.meanDecay[self.numAges-1-k] for k in range(j+1)])) for j in range(self.numAges)) - self.priceDistribution.mean()/(np.multiply.reduce([1-self.meanDecay[self.numAges-1-j] for j in range(self.numAges)]))) for t in range(self.time_steps)) + (gb.quicksum(iss[p,t,t+1]*((self.priceDistribution.mean()-self.price)/(np.multiply.reduce([1-self.meanDecay[t-j] for j in range(t+1)]))) for p in range(self.nProducts) for t in range(self.time_steps-1)))
        self.heuristic_model.setObjective(obj, GRB.MAXIMIZE)

    #map neural network output to issuance decisions for age classes below the lowest target age if blending is allowed
    def get_issuance_blending(self,action, issuance):
        for i in range(self.targetAges[0]-1,-1,-1):
            issuance = [((action[i]+1)/2) * min(self.inventory_position[i],sum(issuance[j-i-1]*self.evaporation_remains_per_age_class[j]*(j-self.targetAges[0]) for j in range(i+1, self.numAges))/((self.targetAges[0]-i)*self.evaporation_remains_per_age_class[i]))] + issuance
        return issuance    

    #allocate issuance volumes to products using allocation LP
    def allocate_issuance(self, issuance):
        self.allocation_model.update()
        for a in range(self.numAges):
            self.allocation_model.remove(self.allocation_model.getConstrByName("issuance_vec"+str(a)))
            iss = [self.allocation_model.getVarByName(f"iss[{p},{a}]") for p in range(self.nProducts)]
            self.allocation_model.addLConstr(gb.quicksum(iss[p] for p in range(self.nProducts)) <= issuance[a], name="issuance_vec"+str(a)) 
        self.allocation_model.update()
        self.allocation_model.optimize()
        issuance_product = {p: {} for p in range(self.nProducts)}
        for p in range(self.nProducts):
            for a in range(self.numAges):
                issuance_product[p][a] = self.allocation_model.getVarByName(f"iss[{p},{a}]").X
        return issuance_product

    def create_allocation_lp(self):
        self.allocation_model = gb.Model()
        #disable printout of Gurobi solution process
        self.allocation_model.setParam('OutputFlag', 0)

        #prepare indices of decision variables
        tuplelist_alloc = [(p,i) for p in range(self.nProducts) for i in range(self.numAges)]
        tuplelist_ff = [(p,l) for p in range(self.nProducts) for l in self.production_levels[p]]

        #decision variables for issuance allocation
        iss = self.allocation_model.addVars(tuplelist_alloc, name="iss")
        #decision variables for production
        ff = self.allocation_model.addVars(tuplelist_ff, lb=[0.0 for i in range(len(tuplelist_ff))], ub=[1.0 for i in range(len(tuplelist_ff))], name="ff")

        #initialize empty issuance vector
        issuance_init = np.zeros(self.numAges)

        #restrict allocation to initial 
        for a in range(self.numAges):
            self.allocation_model.addLConstr(gb.quicksum(iss[p,a] for p in range(self.nProducts)) <= issuance_init[a], name="issuance_vec"+str(a))

        for p in range(self.nProducts):
            #restrict production to quasi-binary
            self.allocation_model.addLConstr(gb.quicksum(ff[p,l] for l in self.production_levels[p]) <= 1.0)
            #relate production to issuance
            self.allocation_model.addLConstr(gb.quicksum(ff[p,l] * l for l in self.production_levels[p]) <= gb.quicksum(iss[p,a] * self.evaporation_remains_per_age_class[a] for a in range(self.numAges)))
            #no blending
            if not self.allowBlending:
                self.allocation_model.addLConstr(gb.quicksum(iss[p,a] for a in range(self.targetAges[p])) <= 0)
            else:
                self.allocation_model.addLConstr(gb.quicksum(iss[p,a] * a * self.evaporation_remains_per_age_class[a] for a in range(self.numAges)) >= self.targetAges[p] * gb.quicksum(iss[p,a] for a in range(self.numAges)))
            #no overage downgrading
            if not self.allowOverage:
                if not p == self.nProducts - 1:
                    self.allocation_model.addLConstr(gb.quicksum(iss[p,a] for a in range(self.targetAges[p+1], self.numAges)) <= 0)

        #objective maximizes expected revenues
        obj = gb.quicksum(ff[p,l] * self.expected_revenue_salvage[p][l] for p in range(self.nProducts) for l in self.production_levels[p])
        self.allocation_model.setObjective(obj, GRB.MAXIMIZE)
                         

    def env_creator(env_config):
        return AmelioratingInventoryEnv(env_config)
    
    #simulator function for evaluating policy
    def simulate_steps(self, nsteps=1, policy=None, plot=False, evaluate=False):
        
        #warm-up
        warm_up_steps = 100
        while(warm_up_steps > 0):
            if self.render_mode == 'human':
                action = np.array([int(i) for i in input("write down action in the format: <purchasing> <production product 1> ... <production product |W|>").split(" ")[:self.nProducts+1]])
                print("action taken:", action)
            else:
                if policy == None:
                    action = self.action_space.sample()
                else:
                    action = policy.compute_single_action(self._get_obs())
            next_state, reward, truncated, done, info = self.step(action) 
            warm_up_steps -= 1 

        self.n_steps = 0
        rewards = np.array([])
        purchasing = np.array([])
        revenues = np.array([])
        purchasing_costs = np.array([])
        decay_salvage = np.array([])
        holding_costs = np.array([])
        prices = np.array([])

        production = {p: np.array([]) for p in range(self.nProducts)}
        issuance = {p: np.array([]) for p in range(self.nProducts)}
        inventories = {a: np.array([]) for a in range(self.numAges)}
        decay_proportions = {a: np.array([]) for a in range(self.numAges)}
        outdating = np.array([])
        iterations = 0
        while(nsteps > 0):
            #print("inventory position: ", self.inventory_position)
            if self.render_mode == 'human':
                action = np.array([int(i) for i in input("write down action in the format: <purchasing> <production product 1> ... <production product |W|>").split(" ")[:self.nProducts+1]])
                print("action taken:", action)
            else:
                if policy == None:
                    action = self.action_space.sample()
                else:
                    action = policy.compute_single_action(self._get_obs(), explore=False) #, explore=False

            #print("TAKEN ACTION: ", action)
            next_state, reward, truncated, done, info = self.step(action) 
            #print(info)
            rewards = np.append(rewards,reward)
            purchasing = np.append(purchasing, info["purchasing"])
            revenues = np.append(revenues, info["revenue"])
            purchasing_costs = np.append(purchasing_costs, info["purchasing_cost"])
            decay_salvage = np.append(decay_salvage, info["decay_salvage"])
            holding_costs = np.append(holding_costs, info["holding_cost"])
            prices = np.append(prices, info["price"])
            for p in range(self.nProducts):
                issuance[p] = np.append(issuance[p], info["issuance"][p])
                production[p] = np.append(production[p], info["production"][p])
            for a in range(self.numAges):
                decay_proportions[a] = np.append(decay_proportions[a], info["decay_proportions"][a])
                inventories[a] = np.append(inventories[a], info["inventory"][a])
            outdating = np.append(outdating, info["outdating"])

            iterations+=1
            if iterations % 500 == 0:
                denormalized_rewards = [((r-self.reward_lb)/(self.reward_ub - self.reward_lb)) * (self.max_reward - self.min_reward) + self.min_reward for r in rewards]
                print(f"{iterations} STEPS SIMULATED")
                print("average reward: ", np.mean(rewards))
                print("average reward w/o normalization: ", np.mean(denormalized_rewards))
                print("90 percent confidence average reward: ", st.norm.interval(0.9, loc=np.mean(denormalized_rewards), scale=np.std(denormalized_rewards)/np.sqrt(len(denormalized_rewards))))
                print("average purchasing: ", np.mean(purchasing))
                print("purchasing variance: ", np.var(purchasing))
                print("average production: ", [np.mean(production[p]) for p in range(self.nProducts)])
                print("mean decay proportion: ", [np.mean(decay_proportions[a]) for a in range(self.numAges)])
                print("average inventory structure: ", [np.mean(inventories[a]) for a in range(self.numAges)])
                print("average outdating: ", np.mean(outdating))
                print("average revenue: ", np.mean(revenues))
                print("average purchasing costs: ", np.mean(purchasing_costs))
                print("average decay_salvage: ", np.mean(decay_salvage))
                print("average holding costs: ", np.mean(holding_costs))
                print("average price: ", np.mean(prices))
                print("price std: ", np.std(prices))
            nsteps-=1

        if plot:
            fig = plt.hist(purchasing)
            plt.show()
            fig = plt.plot(prices)
            plt.show()
        return np.mean(rewards)
    
    def get_heuristic_average(self, final_interval_width=0.4):
        simulate_heuristic_setting = self.simulate_heuristic
        self.simulate_heuristic = True
        interval_width = final_interval_width + 1
        rewards = np.array([])
        while interval_width > final_interval_width:
            for _ in range(500):
                next_state, reward, truncated, done, info = self.step(self.action_space.sample()) 
                rewards = np.append(rewards,reward)
            new_interval = st.norm.interval(0.9,loc=np.mean(rewards),scale=np.std(rewards)/np.sqrt(len(rewards)))
            interval_width = new_interval[1]-new_interval[0]
            print(f"average reward & interval width after {len(rewards)} steps: ", np.mean(rewards), " ", interval_width)

        avg_heuristic_reward = np.mean(rewards)
        print("heuristic average reward: ", avg_heuristic_reward)
        self.simulate_heuristic = simulate_heuristic_setting
        self.reset()

        return avg_heuristic_reward

    def simulate_heuristic_w_cdfs(self, cdfs, price, inventory):
        self.cdf_buffer = cdfs
        simulate_heuristic_setting = self.simulate_heuristic
        start_state = self._get_obs()
        self.n_steps = 0
        self.price = price
        self.inventory_position = inventory
        self.simulate_heuristic = True
        self.use_cdfs_for_regularization = True
        rewards = np.array([])
        for _ in range(self.max_horizon):
            next_state, reward, truncated, done, info = self.step(self.action_space.sample())
            if self.use_adversarial_sampling and self.n_steps == self.adversarial_threshold:
                self.reset()
            rewards = np.append(rewards, reward)
        
        self.simulate_heuristic = simulate_heuristic_setting
        self.use_cdfs_for_regularization = False
        self.n_steps = 0
        self.reset()
        self._get_state_from_obs(start_state)
        mean_reward = np.mean(rewards)

        return mean_reward


    def render_step(self, nsteps=1, policy=None):
        
        fig, axs = plt.subplots(2,2)
        fig.tight_layout(pad=4.0)
        rewards = np.array([])
        while(nsteps > 0):
            axs[1,1].set_visible(False)
            axs[1,0].set_visible(False)
            axs[0,0].clear(); axs[0,1].clear(); axs[1,0].clear(); axs[1,1].clear()
            print(self.inventory_position)
            axs[0,0].bar([i for i in range(1,self.numAges+1)], self.inventory_position, color=['black' if i-1 not in self.targetAges else 'red'for i in range(1,self.numAges+1)])
            axs[0,0].xaxis.set_ticks([i for i in range(1,self.numAges+1)])
            axs[0,0].set_title("inventory position")
            xline = np.linspace(self.priceDistribution.ppf(0.0), self.priceDistribution.ppf(1.0), 100)
            axs[0,1].plot(xline, self.priceDistribution.pdf(xline))
            axs[0,1].get_yaxis().set_visible(False)
            axs[0,1].axvline(self.price, ymax=1, color="red")
            axs[0,1].set_title("price level")

            plt.show(block=False)#
            plt.pause(3)

            if self.render_mode == 'human':
                action = np.array([int(i) for i in input("write down action in the format: <purchasing> <production product 1> ... <production product |W|>").split(" ")[:self.nProducts+1]])
                print("action taken:", action)
            else:
                if policy == None:
                    action = self.action_space.sample()
                else:
                    action = policy.compute_single_action(self._get_obs(), explore=False)
        
            next_state, reward, done, info = self.step(action) 
            print(info)
            print("action taken:", action)
            rewards = np.append(rewards,reward)
            axs[1,0].bar((["purchase"]+[f"produce {y}" for y in range(1,self.nProducts+1)]), action, color=(["green"]+["red" for y in range(1,self.nProducts+1)]))
            axs[1,0].set_title("action")
            axs[1,1].text(0.05,0.01,f"demand: {info['demand']} \n sales: {info['sales']} \n outdating: {info['outdating']} \n overproduction: {info['overproduction']} \n reward: {round(reward, 2)} \n average reward: {round(np.mean(rewards),2)}")
            axs[1,1].set_visible(True)
            axs[1,1].get_yaxis().set_visible(False)
            axs[1,1].get_xaxis().set_visible(False)
            axs[1,1].axis('off')
            axs[1,0].set_visible(True)
            print("average reward: ", np.mean(rewards))
            print("reward std: ", np.std(rewards))

            plt.show(block=False)
            plt.pause(8)
            nsteps-=1
#--------------------------------------#
def upper_bound(env:AmelioratingInventoryEnv, discr_step=0.01):
    #create outer approximation of concave function for expected reward
    tangent_points = {w : [round(i,ndigits=2) for i in np.arange(env.production_levels[w][0],env.production_levels[w][-1]+discr_step/2,discr_step)] for w in range(env.nProducts)}
    slope_tangent_points = {w: [(1-env.demandDistributions[w].cdf(t))*env.brandContributions[w] + env.demandDistributions[w].cdf(t)*env.salvage[w] for t in tangent_points[w]] for w in range(env.nProducts)}
    expected_revenue_tangent_points = {p: {l: env.brandContributions[p] * (sigr.quad(env.ff_function[p], 0, l)[0] + l * (1-env.demandDistributions[p].cdf(l))) + env.salvage[p] * (l*env.demandDistributions[p].cdf(l) - sigr.quad(env.ff_function[p], 0, l)[0]) for l in tangent_points[p]} for p in range(env.nProducts)} 
    break_points = {w: np.concatenate(([tangent_points[w][0]],[(expected_revenue_tangent_points[w][tangent_points[w][i+1]] - expected_revenue_tangent_points[w][tangent_points[w][i]] + slope_tangent_points[w][i]*tangent_points[w][i] - slope_tangent_points[w][i+1]*tangent_points[w][i+1])/(slope_tangent_points[w][i]-slope_tangent_points[w][i+1]) for i in range(len(tangent_points[w])-1)],[tangent_points[w][-1]])) for w in range(env.nProducts)} 
    expected_revenue_break_points = {p: {break_points[p][l]: expected_revenue_tangent_points[p][tangent_points[p][l]] + slope_tangent_points[p][l] * (break_points[p][l] - tangent_points[p][l]) for l in range(len(break_points[p])-1)} for p in range(env.nProducts)} 
    for p in range(env.nProducts):
        expected_revenue_break_points[p][env.sales_bound[p]] = expected_revenue_tangent_points[p][env.sales_bound[p]]

    #discretize price levels
    price_levels_discretized = [i for i in np.arange(env.priceDistribution.ppf(0.0), env.priceDistribution.ppf(1.0), discr_step)]
    price_probabilities_discretized = {i: env.priceDistribution.cdf(i+discr_step) - env.priceDistribution.cdf(i) for i in price_levels_discretized}

    #create upper bound model 
    upper_bound_model = gb.Model()
    #disable printout of Gurobi solution process
    upper_bound_model.setParam('OutputFlag', 0)

    #define decision variables
    ff_indices = [(p,b) for p in range(env.nProducts) for b in break_points[p]]
    ff = upper_bound_model.addVars(ff_indices, name="ff", lb=0.0, ub=1.0)
    purchasing = upper_bound_model.addVars(price_levels_discretized, name="purchasing", lb=0.0, ub=env.maxInventory)
    inv = upper_bound_model.addVars(env.numAges, name="inv", lb=0.0, ub=env.maxInventory)
    iss = upper_bound_model.addVars(env.nProducts, env.numAges, name="iss", lb=0.0, ub=env.maxInventory)
    out = upper_bound_model.addVar(name="out", lb=0.0, ub=env.maxInventory)

    print("Upper Bound Model: Variable Creation Done!")

    #OBJECTIVE
    obj = gb.quicksum(gb.quicksum(ff[p,l] * expected_revenue_break_points[p][l] for l in break_points[p]) for p in range(env.nProducts)) - gb.quicksum(purchasing[p]*p*price_probabilities_discretized[p] for p in price_levels_discretized) + gb.quicksum(inv[a]*(env.decaySalvage[a]*env.meanDecay[a]-env.holdingCosts)/(1-env.meanDecay[a]) for a in range(env.numAges)) - out*env.outdatingCosts
    upper_bound_model.setObjective(obj, GRB.MAXIMIZE)

    print("Upper Bound Model: Objective Added!")

    #CONSTRAINTS
    #inventory balance
    for a in range(1,env.numAges):
        upper_bound_model.addLConstr(inv[a] == (inv[a-1]-gb.quicksum(iss[p,a-1] for p in range(env.nProducts)))*(1-env.meanDecay[a]))
    #inventory in first age class is determined by purchasing behavior
    upper_bound_model.addLConstr(inv[0] == gb.quicksum(purchasing[l] * price_probabilities_discretized[l] * (1-env.meanDecay[0]) for l in price_levels_discretized))
    #outdating is determined by issuance from last age class
    upper_bound_model.addLConstr(out == inv[env.numAges-1] - gb.quicksum(iss[p,env.numAges-1] for p in range(env.nProducts)))
    
    print("Upper Bound Model: Inventory Constraints Added!")

    #production & issuance
    for p in range(env.nProducts):
        #production proportions add up to 1
        upper_bound_model.addLConstr(gb.quicksum(ff[p,l] for l in break_points[p]) == 1)
        #production amounts cannot exceed issuance amounts per product
        upper_bound_model.addLConstr(gb.quicksum(ff[p,l]*l for l in break_points[p]) <= gb.quicksum(iss[p,a]*env.evaporation_remains_per_age_class[a] for a in range(env.numAges)))
        #target ages need to be respected
        upper_bound_model.addLConstr(gb.quicksum(iss[p,a] * a * env.evaporation_remains_per_age_class[a] for a in range(env.numAges)) >= env.targetAges[p] * gb.quicksum(iss[p,a] * env.evaporation_remains_per_age_class[a] for a in range(env.numAges)))
        #prohibit blending
        if not env.allowBlending:
            upper_bound_model.addLConstr(gb.quicksum(iss[p,a] for a in range(env.targetAges[p]-1)) <= 0)
        elif env.blendingRange is not None:
            upper_bound_model.addLConstr(gb.quicksum(iss[p,a] for a in range(env.numAges) if a not in range(env.targetAges[p]-env.blendingRange,env.targetAges[p]+env.blendingRange+1)) <= 0)

    print("Upper Bound Model: Product Constraints Added!")

    upper_bound_model.optimize()

    print("OPTIMAL INVENTORY STRUCTURE: ",[upper_bound_model.getVarByName(f"inv[{i}]").X for i in range(env.numAges)])
    print("OPTIMAL ISSUANCE VOLUMES: ",  [sum(upper_bound_model.getVarByName(f"iss[{p},{a}]").X for p in range(env.nProducts)) for a in range(env.numAges)])
    print("OPTIMAL PRODUCTION VOLUMES: ",  [sum(upper_bound_model.getVarByName(f"iss[{p},{a}]").X * env.evaporation_remains_per_age_class[a] for a in range(env.numAges)) for p in range(env.nProducts)])
    print("OPTIMAL PURCHASING: ", sum(upper_bound_model.getVarByName(f"purchasing[{l}]").X * price_probabilities_discretized[l] for l in price_levels_discretized))

    print("OPTIMAL PURCHASING COST: ", sum(upper_bound_model.getVarByName(f"purchasing[{l}]").X * l * price_probabilities_discretized[l] for l in price_levels_discretized))
    print("OPTIMAL EXPECTED REVENUES: ", sum(upper_bound_model.getVarByName(f"ff[{p},{l}]").X * expected_revenue_break_points[p][l] for p in range(env.nProducts) for l in break_points[p]))
    print("OPTIMAL HOLDING COSTS: ", sum(upper_bound_model.getVarByName(f"inv[{a}]").X * env.holdingCosts/(1-env.meanDecay[a]) for a in range(env.numAges)))
    print("OPTIMAL SALVAGE DECAY: ", sum(upper_bound_model.getVarByName(f"inv[{a}]").X * (env.decaySalvage[a]*env.meanDecay[a])/(1-env.meanDecay[a]) for a in range(env.numAges)))

    return upper_bound_model.ObjVal, np.array([upper_bound_model.getVarByName(f"inv[{i}]").X for i in range(env.numAges)])