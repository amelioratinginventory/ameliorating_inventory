#--------------------------------------#
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import ray
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import string
import pprint
from pathlib import Path

import tensorflow as tf
import argparse
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from numpy import inf
import scipy.integrate as sigr
import ray.rllib.algorithms.apo as apo
import ray.rllib.algorithms.ppo as ppo 
from ray import air, tune
from ray.tune.registry import register_env, register_trainable
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import ResultGrid

from ray.rllib.utils import try_import_tf
from ray.rllib.utils.schedules.polynomial_schedule import PolynomialSchedule
from ray.tune.logger import pretty_print
from ray.air.config import CheckpointConfig
from AmelioratingInventoryPOM import AmelioratingInventoryEnv as env
from AmelioratingInventoryPOM import upper_bound as ub_function
from ray.rllib.algorithms.apo.apo_tf_policy import APOTF2Policy
from ray.rllib.utils.checkpoints import get_checkpoint_info
from stochastic.processes.diffusion import vasicek as vsk

import wandb
import openpyxl
import os

#----------------------------------------#
def data_from_xlsx_named_range(xlsx_file, range_name):
   cell_name = xlsx_file.defined_names[range_name].value
   ws, reg = cell_name.split('!')
   if ws.startswith("'") and ws.endswith("'"):
      ws = ws[1:-1]
   region = xlsx_file[ws][reg]
   data = [cell.value for row in region for cell in row]
   return data

#------------------------------------------#

def main():
   
   #disable GPU for training
   #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

   problem_id = "spirits_012_n"
   storage_path = f"{os.getcwd()}/problem_configurations/{problem_id}/training_results/"

   #ray training settings
   horizon = 5250
   truncation_length = 250
   num_workers = 50
   minibatch_size = 0.1
   recovery_checkpoint = None #f"{os.getcwd()}/problem_configurations/{problem_id}/training_results/full_blending_py01_nn_xy2_lp_50w/checkpoint_000336"
   start_iteration = 0 if recovery_checkpoint is None else int(recovery_checkpoint[-3:])-1

   if recovery_checkpoint is not None:
      run_id = recovery_checkpoint.split("training_results/")[1].split("/")[0]
      checkpoint_dir = recovery_checkpoint.split("checkpoint_")[0]
   use_common_random_numbers = False
   training_iterations = 500
   simulate_heuristic_for_eval = False

   #load problem configuration given problem_id
   path_config = Path(f"{os.getcwd()}\\problem_configurations\\{problem_id}\\config.json") 
   if path_config.is_file():
      with open(path_config, 'r') as f:
         problem_config = json.load(f)
      allowBlending = problem_config["allowBlending"]
      numAges = problem_config["numAges"]
      nProducts = problem_config["nProducts"]
      targetAges = problem_config["targetAges"]
      ageRange = problem_config["ageRange"]
      maxInventory = problem_config["maxInventory"]
      evaporation = problem_config["evaporation"]
      demand_means = problem_config["demand_means"]
      demand_covs = problem_config["demand_covs"]
      price_mean = problem_config["price_mean"]
      price_std = problem_config["price_std"]
      price_truncation = problem_config["price_truncation"]
      price_alpha = problem_config["price_alpha"] if "price_alpha" in problem_config else None
      price_beta = problem_config["price_beta"] if "price_beta" in problem_config else None
      decay_mean = problem_config["decay_mean"]
      decay_cov = problem_config["decay_cov"]
      correlation_demand_salesprice = problem_config["correlation_demand_salesprice"]
      sales_means = problem_config["sales_means"]
      sales_covs = problem_config["sales_covs"]
      use_price_process = problem_config["use_price_process"]
      price_speed = problem_config["price_speed"]
      holding_cost = problem_config["holdingCosts"]
      decay_salvage = problem_config["decaySalvage"]
      salvage_range = problem_config["salvage_range"]
      production_step_size = problem_config["production_step_size"]
      min_ppf = problem_config["min_ppf"]
      max_ppf = problem_config["max_ppf"]
      expected_revenue = {p: {float(l): problem_config["expected_revenue"][str(p)][str(l)] for l in problem_config["expected_revenue"][str(p)]} for p in range(nProducts)}
      slope = {p: {float(l): problem_config["slope"][str(p)][str(l)] for l in problem_config["slope"][str(p)]} for p in range(nProducts)}
      upper_bound = {"max_reward": problem_config["upper_bound"]["max_reward"], "inventory_position": np.array(problem_config["upper_bound"]["inventory_position"])}
   else:
      #set problem configuration
      # problem size parameters
      numAges = 10
      nProducts = 3
      if problem_id[-1] == "n":
         targetAges = [3,4,5]
      else:
         targetAges = [2,4,6]
      # ageRange = [[i for i in range(targetAges[p],targetAges[p+1])] for p in range(nProducts-1)]
      # ageRange.append([i for i in range(targetAges[-1],numAges)])
      ageRange = None
      if problem_id[8] == "0":
         maxInventory = 50
      elif problem_id[8] == "1":
         maxInventory = 30
      if problem_id[9] == "0":
         evaporation = 0.03
      elif problem_id[9] == "1":
         evaporation = 0.02

      # further parameters   
      price_mean = 200.0
      
      if problem_id[10] == "0":
         price_std = 50.0
         price_truncation = 70.0
      else:
         price_std = 30.0
         price_truncation = 50.0
      price_family = "normal"
      price_alpha = 2.5 if price_family == "gamma" else None
      price_beta = 20.0 if price_family == "gamma" else None
      correlation_demand_salesprice = [0.5 for _ in range(nProducts)]
      demand_means = [10.0, 7.0, 5.0]
      demand_covs = [0.25, 0.25, 0.25]
      sales_means = [250,350,500]
      sales_covs = [0.1,0.1,0.1]
      salvage_range = [0.4,0.6]
      holding_cost = 2.5
      decay_salvage = [20.0 + (i*5.0) for i in range(numAges)]
      decay_mean = [0.03 * (0.95**i) for i in range(numAges)] 
      decay_cov = [0.8 for i in range(numAges)]
      use_price_process = False
      price_speed = 0.1 if use_price_process else None
      min_ppf = 1e-12
      max_ppf = 1-1e-12
      production_step_size = 0.01
      upper_bound = None
      if problem_id[-1] == "n":
         if problem_id[-3] == "1":
            allowBlending = False
         elif problem_id[-3] == "2":
            allowBlending = True
      else:
         if problem_id[-1] == "1":
            allowBlending = False
         elif problem_id[-1] == "2":
            allowBlending = True

   #uncertainty distribution parameters
   demandDistributions = [st.norm(loc=demand_means[i], scale=demand_covs[i]*demand_means[i]) for i in range(nProducts)]
   salesPriceDistributions = [st.norm(loc=sales_means[i], scale=sales_covs[i]*sales_means[i]) for i in range(nProducts)]
   correlation_demand_salesprice = [0.5 for _ in range(nProducts)]
   cov_dem_sp = [correlation_demand_salesprice[p]*demandDistributions[p].std()*salesPriceDistributions[p].std() for p in range(nProducts)]
   demand_salesprice_distribution = {p: st.multivariate_normal([demandDistributions[p].mean(), salesPriceDistributions[p].mean()], [[demandDistributions[p].var(), cov_dem_sp[p]],[cov_dem_sp[p], salesPriceDistributions[p].var()]]) for p in range(nProducts)} # type: ignore
   salvage = {p: lambda d, gamma, p=p: salvage_range[0] + ((demandDistributions[p].cdf(d) - salesPriceDistributions[p].cdf(gamma) + 1)/2) * (salvage_range[1]-salvage_range[0]) for p in range(nProducts)}
   priceDistribution = st.truncnorm(loc=price_mean, scale=price_std, a=(-price_truncation/price_std), b=(price_truncation/price_std))
   price_process = vsk.VasicekProcess(speed=price_speed, mean=price_mean, vol=price_std*np.sqrt(2*price_speed), t=1) if use_price_process else None
   
   #preprocessing: expected revenue calculation
   products = [i for i in range(nProducts)]
   production_levels = {p: [round(i,2) for i in np.arange(0,demandDistributions[p].ppf(max_ppf)+production_step_size,production_step_size)] for p in range(nProducts)}
   production_step_size_lp = 0.1
   
   def expected_revenue_function(p: int, x: float):
      x = min(demandDistributions[p].ppf(max_ppf), x)
      return sigr.dblquad(lambda d, gamma: demand_salesprice_distribution[p].pdf([d,gamma]) * gamma * (d + (x-d) * salvage[p](d,gamma)), salesPriceDistributions[p].ppf(min_ppf), salesPriceDistributions[p].ppf(max_ppf), 0, x)[0] + x * sigr.dblquad(lambda d, gamma: demand_salesprice_distribution[p].pdf([d,gamma]) * gamma, salesPriceDistributions[p].ppf(min_ppf), salesPriceDistributions[p].ppf(max_ppf), x, demandDistributions[p].ppf(max_ppf))[0] 
   
   def slope_function(p: int, x: float):
      if x < demandDistributions[p].ppf(max_ppf):
         return sigr.dblquad(lambda d, gamma: demand_salesprice_distribution[p].pdf([d,gamma])*gamma, salesPriceDistributions[p].ppf(min_ppf), salesPriceDistributions[p].ppf(max_ppf), x, demandDistributions[p].ppf(max_ppf))[0] + sigr.dblquad(lambda d, gamma: demand_salesprice_distribution[p].pdf([d,gamma])*salvage[p](d,gamma)*gamma, salesPriceDistributions[p].ppf(min_ppf), salesPriceDistributions[p].ppf(max_ppf), 0, x)[0]
      else:
         return 0

   
   if not path_config.is_file():
      exp_rev_path = Path(f"{os.getcwd()}\\problem_configurations\\{problem_id}\\expected_revenue.json") 
      if not exp_rev_path.is_file():
         expected_revenue = {p: {l: 0 for l in production_levels[p]} for p in range(nProducts)}
         slope = {p: {l: 0 for l in production_levels[p]} for p in range(nProducts)}
         for p in range(nProducts):
            for l in production_levels[p]:
               expected_revenue[p][l] = expected_revenue_function(p,l)
               slope[p][l] = slope_function(p,l)
               print(f"PRODUCT: {p}, LEVEL: {l}, EXP_REV: {expected_revenue[p][l]}, SLOPE: {slope[p][l]}")   
      else:
         with open(exp_rev_path, 'r') as f:
            res = json.load(f)
         expected_revenue = {p: {l: res["expected_revenue"][str(p)][str(l)] for l in production_levels[p]} for p in range(nProducts)}
         slope = {p: {l: res["slope"][str(p)][str(l)] for l in production_levels[p]} for p in range(nProducts)}

   evaporation_remains_per_age_class = {i: (1-evaporation)**(i+1) for i in range(numAges)}
   acc_inv_target = {p: 1/np.multiply.reduce([1-decay_mean[targetAges[p]-k] for k in range(targetAges[p]+1)]) for p in products}
   acc_c_target = {p: sum((-decay_salvage[targetAges[p]-j]*decay_mean[targetAges[p]-j]+holding_cost)/(np.multiply.reduce([1-decay_mean[targetAges[p]-k] for k in range(j+1)])) for j in range(targetAges[p]+1)) for p in products}

   def conditional_demand_distribution(p, gamma):
        return st.norm(loc=demand_means[p] + (demand_covs[p]*demand_means[p])/sales_means[p] * correlation_demand_salesprice[p]*(gamma - sales_means[p]), scale=np.sqrt(1-correlation_demand_salesprice[p]**2)*(demand_covs[p]*demand_means[p]))

   def get_critical_ratio(p, gamma, price):
        c_u = max(0,gamma*evaporation_remains_per_age_class[targetAges[p]] - price*acc_inv_target[p] - acc_c_target[p])
        c_o = price*acc_inv_target[p] + acc_c_target[p] - salvage[p](demand_means[p],gamma) * gamma * evaporation_remains_per_age_class[targetAges[p]]
        critical_ratio = max(min_ppf,min(c_u/(c_u+c_o),max_ppf))
        return max(0,conditional_demand_distribution(p,gamma).ppf(critical_ratio))

   def get_production_targets(p, price=None):
        if price is None:
            price = price_mean
        #get the demand distribution for the product given a certain gamma level
        return sigr.quad(lambda gamma: get_critical_ratio(p,gamma,price) * salesPriceDistributions[p].pdf(gamma), salesPriceDistributions[p].ppf(min_ppf), salesPriceDistributions[p].ppf(max_ppf))[0]
      
   #calculate the order up to targets for each product and price level and store them in a json file
   path = Path(f"{os.getcwd()}\\problem_configurations\\{problem_id}\\production_targets.json")
   if not path.is_file():
      production_targets = {}
      for price in np.arange(0.000,1.001,0.001):
         price = round(price,ndigits=3)
         print(price)
         production_targets[round(price,ndigits=3)] = [get_production_targets(p,priceDistribution.ppf(price)) for p in products]
         purchase = sum([production_targets[price][p] * acc_inv_target[p]/evaporation_remains_per_age_class[targetAges[p]] for p in products])
         if purchase > maxInventory:
            for p in products:
               production_targets[price][p] = production_targets[price][p] * maxInventory/purchase
            purchase = maxInventory
         production_targets[price].append(purchase)
         load_data = {"targets": production_targets}
      with open(path, 'w') as f:
         json.dump(load_data, f)
      newsvendor_prod = None
   else:
      with open(path, 'r') as f:
         load_data = json.load(f)
      production_targets = {float(k): load_data["targets"][k] for k in load_data["targets"].keys()}
      if "newsvendor_prod" in load_data:
         newsvendor_prod = {p: {round(float(l),ndigits=2): load_data["newsvendor_prod"][f"{p}"][l] for l in sorted(load_data["newsvendor_prod"][f"{p}"].keys())} for p in products}
      else:
         newsvendor_prod = None

   use_issuance_model = True
  
   AIE_config = {"numAges":numAges, "nProducts":nProducts, "targetAges":targetAges, "maxInventory":maxInventory, "evaporation":evaporation, 
    	         "demandDistributions":demandDistributions, "priceDistribution": priceDistribution, "decay_mean": decay_mean, "decay_cov": decay_cov, "demand_salesprice_distribution": demand_salesprice_distribution,
               "salesPriceDistributions":salesPriceDistributions, "correlation_demand_salesprice":correlation_demand_salesprice, "brandContributions":sales_means,
               "holdingCosts":holding_cost, "decaySalvage":decay_salvage, "salvage":salvage, "expected_revenue":expected_revenue, "slope": slope,
               "min_ppf":min_ppf, "max_ppf":max_ppf, "production_step_size":production_step_size, "production_step_size_lp":production_step_size_lp, "upper_bound":upper_bound,
               "allowBlending": allowBlending, "blendingRange":None, "ageRange":ageRange, "drl_for_production": False, "products_using_drl":None, "priceProcess": price_process, 
               "action_space_design":"box_continuous", "use_adversarial_sampling":False, "render_mode":'rgb_array', "horizon":horizon, "simulate_heuristic":False, "use_common_random_numbers": use_common_random_numbers,
               "reward_lb":-1.0, "reward_ub":1.0, "use_issuance_model":use_issuance_model, "penalty_structure":3.0, "history_length":numAges, "penalty_heuristic_deviation":0, "production_targets":production_targets, "newsvendor_prod":newsvendor_prod}  

   ray.init(num_cpus=num_workers+1, num_gpus=0)
   
   eval_runs = 10
   eval_length = 500
   if recovery_checkpoint is not None:
      try:
         with open(checkpoint_dir+"best_checkpoint.json", 'r') as f:
            cdfs = json.load(f)["eval_buffer"]   # Attempt to load JSON data
      except FileNotFoundError:
         print(f"Error: The file does not exist.")
         cdfs = None  # Default to an empty dictionary
      except json.JSONDecodeError:
         print(f"Warning: The file is empty or contains invalid JSON.")
         data = None  # Default to an empty dictionary
      if cdfs is None:
         eval_buffer = {i: {j: np.random.rand(numAges+1).tolist() for j in range(eval_length)} for i in range(eval_runs)}
      else:
         eval_buffer = {i: {j: cdfs[str(i)][str(j)] for j in range(eval_length)} for i in range(eval_runs)}
   else:
      eval_buffer = {i: {j: np.random.rand(numAges+1).tolist() for j in range(eval_length)} for i in range(eval_runs)}

   register_env("AmelioratingInventory", lambda config: env(config))
   test_env = env(AIE_config)

   if "newsvendor_prod" not in load_data:
      newsvendor_prod = {p: {} for p in products}
      for v in np.arange(0.00,maxInventory+0.01,0.01):
         if v%5 == 0:
            print(v)
         for p in products[:-1]:
            test_env.newsvendor_model_prod[p].update()
            test_env.newsvendor_model_prod[p].getConstrByName("start_inv").rhs = v
            test_env.newsvendor_model_prod[p].update()
            test_env.newsvendor_model_prod[p].optimize()
            newsvendor_prod[p][v] = sum(test_env.newsvendor_model_prod[p].getVarByName(f"production[{p},{l}]").X * l for l in test_env.production_levels[p]) / test_env.evaporation_remains_per_age_class[test_env.targetAges[p]]
      newsvendor_prod[nProducts-1] = {v: v for v in np.arange(0.00,maxInventory+0.01,0.01)}
      load_data["newsvendor_prod"] = newsvendor_prod
      with open(f"{os.getcwd()}\\problem_configurations\\{problem_id}\\production_targets.json", 'w') as f:
         json.dump(load_data, f) 

   #raise ValueError("NEWSVENDOR PRODUCTION CALCULATED")
   
   ray.rllib.utils.check_env(test_env)

   eval_starting_price, eval_starting_inv = test_env.simulate_starting_state_eval(50)
   
   if simulate_heuristic_for_eval:
      eval_heuristic = []
      for i in range(eval_runs):
         eval_heuristic += [test_env.simulate_w_cdfs(cdfs=eval_buffer[i], policy=None, initial_price=eval_starting_price, initial_inventory=eval_starting_inv)[1]]

      mean_eval_heuristic = np.mean(eval_heuristic)
      print("HEURISTIC AVERAGE EVAL: ", mean_eval_heuristic)     
   else:
      mean_eval_heuristic = 0 

   path_ub = Path(f"{os.getcwd()}\\problem_configurations\\{problem_id}\\upper_bound.json")
   if not path_ub.is_file():
      ub = ub_function(test_env, discr_step=production_step_size)
      with open(path_ub, 'w') as f:
         json.dump(ub, f)
      upper_bound = {"max_reward": ub["max_reward"], "inventory_position": np.array(ub["inventory_position"])}
   if not path_config.is_file():
      with open (path_ub, 'r') as f:
         res = json.load(f)
      ub_json = {"max_reward": res["max_reward"], "inventory_position": res["inventory_position"]}
      JSON_config = {"allowBlending":allowBlending, "numAges":numAges, "nProducts":nProducts, "targetAges":targetAges, "ageRange":ageRange, "maxInventory":maxInventory, "evaporation":evaporation, 
         "demand_means":demand_means, "demand_covs":demand_covs, "price_mean": price_mean, "price_std": price_std, "price_truncation":price_truncation, "price_alpha":price_alpha, "price_beta":price_beta, "use_price_process":use_price_process, "price_speed":price_speed, "decay_mean": decay_mean, "decay_cov": decay_cov,
         "correlation_demand_salesprice":correlation_demand_salesprice, "sales_means":sales_means, "sales_covs":sales_covs, 
         "holdingCosts":holding_cost, "decaySalvage":decay_salvage, "salvage_range":salvage_range, "expected_revenue":expected_revenue, "slope": slope,
         "min_ppf":min_ppf, "max_ppf":max_ppf, "production_step_size":production_step_size, "upper_bound":ub_json}
      print("WRITE TO CONFIG FILE")
      with open(path_config, 'w') as f:
         json.dump(JSON_config, f)

   register_trainable("APO", apo.APO)
   assert tf.executing_eagerly()

   #restore algorithm from checkpoint if required
   if recovery_checkpoint is not None:
      algo = Algorithm.from_checkpoint(recovery_checkpoint)
      with open(checkpoint_dir+"best_checkpoint.json", 'r') as f:
         checkpoint_stats = json.load(f)
      nCheckpoints = 3
      best_average_reward = checkpoint_stats["avg_reward"] if "avg_reward" in checkpoint_stats else [(-np.Inf, None) for _ in range(nCheckpoints)]
      best_reward_estimate = checkpoint_stats["reward_estimate"] if "reward_estimate" in checkpoint_stats else [(-np.Inf, None) for _ in range(nCheckpoints)]
      best_eval = checkpoint_stats["eval"] if "eval" in checkpoint_stats else [(-np.Inf, None) for _ in range(nCheckpoints)]
      #CHANGE THIS WHEN CHECKPOINTING
      average_reward_estimate_checkpoint = 0.91125
      bias_estimate_checkpoint = -0.039836
      def update_apo_estimates(w):
         for k in w.policy_map.keys():
            # print("UPDATING WORKER")
            # print(w)
            # print(w.policy_map[k].average_reward_estimate)
            w.policy_map[k].average_reward_estimate = average_reward_estimate_checkpoint
            w.policy_map[k].bias_estimate = bias_estimate_checkpoint
            #print(w.policy_map[k].average_reward_estimate)
      
      algo.workers.foreach_worker(
        func=update_apo_estimates
      )
      for k in algo.workers.local_worker().policy_map.keys():
         print(f"LOCAL WORKER AVERAGE REWARD {k}: ", algo.workers.local_worker().get_policy(k).average_reward_estimate)
      print(f"RECOVERED ALGORITHM from checkpoint {recovery_checkpoint}")
      print("START ITERATION: ", start_iteration)
   else:
      init_average_reward_estimate = test_env.simulate_n_steps(100, None, plot = False, warm_up = 20)
      print("initial average reward estimate from random policy: ", init_average_reward_estimate)

      adv_sampl = AIE_config["use_adversarial_sampling"] if "use_adversarial_sampling" in AIE_config else False
      #heuristic_average = test_env.get_heuristic_average(final_interval_width = 0.02)

      use_bias_normalization = False
      if use_bias_normalization:
         heuristic_average = test_env.get_heuristic_average()
      else:
         heuristic_average = None

      config = apo.APOConfig().environment("AmelioratingInventory", env_config=AIE_config)
      #config = ppo.PPOConfig().environment("AmelioratingInventory", env_config=AIE_config)
      config.reporting(metrics_num_episodes_for_smoothing=num_workers*10)
      config.rollouts(num_rollout_workers=num_workers, rollout_fragment_length='auto', batch_mode="complete_episodes")
      config.framework("tf2")
      config.training(lr=7e-5, model={"vf_share_layers": False, "fcnet_hiddens": [128,128]}, use_gae = True, lambda_=0.93, gamma=1.0, sgd_minibatch_size = int(minibatch_size*num_workers*horizon), num_sgd_iter=30, apo_step_size=0.2, bias_factor=0.4, use_bias_normalization = use_bias_normalization, heuristic_average=heuristic_average, init_average_reward_estimate = init_average_reward_estimate, shuffle_sequences = True, train_batch_size = num_workers*horizon, truncation_length=truncation_length, clip_param=0.2)
      #config.training(lr=7e-5, model={"vf_share_layers": False, "fcnet_hiddens": [64,64]}, use_gae = True, lambda_=0.9, gamma=0.99, sgd_minibatch_size = int(minibatch_size*num_workers*horizon), num_sgd_iter=30, shuffle_sequences = True, train_batch_size = num_workers*horizon, truncation_length=truncation_length, clip_param=0.2)
      blending_setting = "full" if AIE_config["blendingRange"] is None else AIE_config["blendingRange"]
      nn_setting = "full" if not use_issuance_model else "p" if not AIE_config["drl_for_production"] else "py" if all([p in AIE_config["products_using_drl"] for p in range(nProducts)]) else "py"+"".join([str(p) for p in AIE_config["products_using_drl"]]) 
      lp_setting = "none" if not use_issuance_model else "x" if AIE_config["drl_for_production"] and all([p in AIE_config["products_using_drl"] for p in range(nProducts)]) else "xy" if not AIE_config["drl_for_production"] else "xy"+"".join([str(p) for p in range(nProducts) if p not in AIE_config["products_using_drl"]])
      run_id = f"{blending_setting}_blending_{nn_setting}_nn_{lp_setting}_lp_{num_workers}w_times"
      
      print("RUN ID: ", run_id)

      param_space_config = config.to_dict()
      algo = config.build()
      nCheckpoints = 3
      best_average_reward = [(-np.Inf, None) for _ in range(nCheckpoints)]
      best_reward_estimate = [(-np.Inf, None) for _ in range(nCheckpoints)]
      best_eval = [(-np.Inf, None) for _ in range(nCheckpoints)]
      
   wandb.init(project=problem_id, name=run_id)
   if not os.path.isdir(storage_path+run_id):
      os.mkdir(storage_path+run_id)

   #track sample and training times
   times = {"sample_time": [], "train_time": [], "lp_time": [], "env_time": [], "nn_time": []}

   for i in range(start_iteration, training_iterations):
      print(f"TRAINING ITERATION {i}")
      train_results = algo.train()
      print(train_results["info"])
      times["sample_time"].append(train_results["info"]["learner"]["sample_time"])
      times["train_time"].append(train_results["info"]["learner"]["train_time"])  
      logger_dict = {}
      logger_dict["episode_reward_mean"] = train_results["episode_reward_mean"]
      if run_id.split("_")[-1] not in ["ppo", "sac"]:
         logger_dict["average_reward_estimate"] = train_results["info"]["learner"]["default_policy"]["learner_stats"]["average_reward_estimate"]
         logger_dict["bias_estimate"] = train_results["info"]["learner"]["default_policy"]["learner_stats"]["bias_estimate"]
         logger_dict["purchasing_variance"] = train_results["info"]["learner"]["default_policy"]["learner_stats"]["purchasing_var"]
      logger_dict["rewards"] = train_results["hist_stats"]["episode_reward"]
      logger_dict["entropy"] = train_results["info"]["learner"]["default_policy"]["learner_stats"]["entropy"]
      logger_dict["kl"] = train_results["info"]["learner"]["default_policy"]["learner_stats"]["kl"]
      logger_dict["vf_explained_var"] = train_results["info"]["learner"]["default_policy"]["learner_stats"]["vf_explained_var"]
      logger_dict["total_loss"] = train_results["info"]["learner"]["default_policy"]["learner_stats"]["total_loss"]
      logger_dict["policy_loss"] = train_results["info"]["learner"]["default_policy"]["learner_stats"]["policy_loss"]
      logger_dict["vf_loss"] = train_results["info"]["learner"]["default_policy"]["learner_stats"]["vf_loss"]
      wandb.log(logger_dict)
      #wandb.log({"episode_reward_mean":episode_reward_mean, "rewards":rewards, "entropy":entropy, "kl":kl, "vf_explained_var":vf_explained_var, "total_loss":total_loss, "policy_loss":policy_loss, "vf_loss":vf_loss})
      if i>start_iteration and (logger_dict["episode_reward_mean"] > best_average_reward[0][0]):
         if best_average_reward[0][1] is not None and best_average_reward[0][1] not in [best_reward_estimate[i][1] for i in range(nCheckpoints)] and best_average_reward[0][1] not in [best_eval[i][1] for i in range(nCheckpoints)]:
            algo.delete_checkpoint(best_average_reward[0][1])
         best_average_reward[0] = (logger_dict["episode_reward_mean"], checkpoint)
         best_average_reward.sort(key=lambda x: x[0])
      if i>start_iteration and run_id.split("_")[-1] not in ["ppo", "sac"] and logger_dict["average_reward_estimate"] > best_reward_estimate[0][0]:
         if best_reward_estimate[0][1] is not None and best_reward_estimate[0][1] not in [best_average_reward[i][1] for i in range(nCheckpoints)] and best_reward_estimate[0][1] not in [best_eval[i][1] for i in range(nCheckpoints)]:
            algo.delete_checkpoint(best_reward_estimate[0][1])
         best_reward_estimate[0] = (logger_dict["average_reward_estimate"], checkpoint)
         best_reward_estimate.sort(key=lambda x: x[0])
      #delete checkpoint if none of the both outer if-loops is true
      if run_id.split("_")[-1] not in ["ppo", "sac"]:
         if i>start_iteration and (logger_dict["episode_reward_mean"] < best_average_reward[0][0] and logger_dict["average_reward_estimate"] < best_reward_estimate[0][0]):
            if checkpoint_deletable and checkpoint not in [best_average_reward[i][1] for i in range(nCheckpoints)] and checkpoint not in [best_reward_estimate[i][1] for i in range(nCheckpoints)]:
               algo.delete_checkpoint(checkpoint)
      else: 
         if i>start_iteration and (logger_dict["episode_reward_mean"] < best_average_reward[0][0]):
            if checkpoint_deletable and checkpoint not in [best_average_reward[i][1] for i in range(nCheckpoints)]:
               algo.delete_checkpoint(checkpoint)
      #save checkpoint
      checkpoint = algo.save(storage_path+run_id)
      checkpoint_deletable = True
      #evaluate policy every 5 times and save best checkpoint
      if i>=100 and i%5 == 0:
         eval_algo = []
         for j in range(eval_runs):
            eval_algo += [test_env.simulate_w_cdfs(cdfs=eval_buffer[j], policy=algo, initial_price=eval_starting_price, initial_inventory=eval_starting_inv)[1]]
         eval_mean = np.mean(eval_algo)
         print(f"ITERATION {i}, EVALUATION: {eval_mean}")
         print(f"EVAL HEURISTIC: {mean_eval_heuristic}")
         if eval_mean > best_eval[0][0]:
            best_eval[0] = (eval_mean, checkpoint)
            best_eval.sort(key=lambda x: x[0])
            checkpoint_deletable = False
            if best_eval[0][1] is not None and best_eval[0][1] not in [best_average_reward[i][1] for i in range(nCheckpoints)] and best_eval[0][1] not in [best_reward_estimate[i][1] for i in range(nCheckpoints)]:
               algo.delete_checkpoint(best_eval[0][1])
         if eval_mean > mean_eval_heuristic:
            print("EVALUATION BETTER THAN HEURISTIC")
         
      with open(storage_path+run_id+"/best_checkpoint.json", 'w') as f:
         json.dump({"reward_estimate": best_reward_estimate, "avg_reward": best_average_reward, "eval": best_eval, "eval_heuristic":mean_eval_heuristic, "eval_buffer": eval_buffer}, f)
   #get lp_time, env_time, and nn_time 
   record_times = test_env.simulate_record_times(algo, n_steps=500)
   times["env_time"] = record_times["env_time"].tolist()
   times["lp_time"] = record_times["lp_time"].tolist()
   times["nn_time"] = record_times["nn_time"].tolist()

   algo.stop()
   # best_result = results.get_best_result(metric="episode_reward_mean", scope="all")
   # best_trial = results._experiment_analysis.get_best_trial(metric="info/learner/default_policy/learner_stats/average_reward_estimate", scope="all")
   # best_checkpoint = results._experiment_analysis.get_best_checkpoint(best_trial, metric ="info/learner/default_policy/learner_stats/average_reward_estimate")

   

   #store times in json
   with open(storage_path+run_id+"/times_2.json", 'w') as f:
      json.dump(times, f)
   
   print("CHECKPOINT: ", best_reward_estimate)
  
   data_size = 50_000

   if best_reward_estimate is not None:
      algorithm_path = best_eval[-1][1] 
      checkpoint_info = get_checkpoint_info(algorithm_path)
      #raise ValueError(f"Checkpoint info: {checkpoint_info}")
      state = Algorithm._checkpoint_info_to_algorithm_state(
         checkpoint_info = checkpoint_info,
         policy_ids = None,
         policy_mapping_fn=None,
         policies_to_train=None,
      )
      state["config"]["num_workers"] = 1
      policy = Algorithm.from_state(state)
      print("Policy loaded")
      test_env = env(AIE_config)
      features, responses, rewards = test_env.simulate_data_for_regression(policy, data_size=data_size)
      regression_data = {"features":features.tolist(), "responses":responses.tolist(), "rewards": rewards.tolist()}
      with open(storage_path+run_id+"/regression_data.json", 'w') as f:
               json.dump(regression_data, f)
 
#----------------------------------------#
if __name__ == '__main__':
	main()
