
import os
import sys
import time
import math
import copy
import random
import statistics as stat
import numpy as np
import scipy
from scipy import linalg
from scipy.linalg import null_space
from scipy.optimize import lsq_linear


''' Parameters'''
simulationtime = 10 # number of independent runs
num_system = 4 # number of subsystems
min_redundant_level = 2# lower bound of n
max_redundant_level = 10# upper bound of n

MaxCycle = 100# maximal iteration
BeeSize = 100# population size
Limit = 10 # the threshold for using scout


    
''' Contents used for result storage'''
The_final_sol_brief = [[] for _ in range(simulationtime)] #store the information of the resource allocated
The_final_objective_constraints =  [[] for _ in range(simulationtime)] # store the objective and constraints (cost and weight)
The_availability_final =  [[] for _ in range(simulationtime)]
The_cycle_sol =  [[] for _ in range(simulationtime)]
    
for run in range(simulationtime):
    tStart = time.time()
    
    '''data inputs'''
    cost_given = [6.3, 4.1, 3.9, 2.7] # system cost limit for each subsys
    weight_given = [9,4,4,3] # system wight limit for each subsys
    mu = [2.42,1.63,2.45,3.51]# repaire rate for each subsys
    lamda_s = [0.082,0.061,0.110,0.093]# standby_failure_rate for each subsys
    lamda_w = [0.140,0.092,0.104,0.083]# working_failure_rate for each subsys
    switch_rate = [0.974,0.981,0.993,0.959]# switching_rate for each subsys
    k_value = [2,1,3,2]
   

    C_limit = 80 # system cost limit
    W_limit = 100# system weight limit
    
    num = [0 for i in range(BeeSize)]
    global_best = []    
    
    
    
    '''      Functions      '''
    
    def generate_random_cutpoints(a, b, N, variation=3):
        # 生成均勻分布的 N 個點
        cutpoints = np.linspace(a, b, N+2)[1:-1]
        
        # 將每個點加上一點隨機變動，避免過於固定
        random_offsets = np.random.randint(-variation, variation+1, size=N)
        
        # 加上隨機變動後四捨五入取整數
        cutpoints = [int(round(point + offset)) for point, offset in zip(cutpoints, random_offsets)]
        
        # 保證 cutpoints 不會超出區間 [a, b]，並且確保每個點是唯一的
        cutpoints = sorted(max(a, min(b, x)) for x in cutpoints)
        
        return cutpoints 


    def generate_n_k(min_redundant_level, max_redundant_level, k):
        while True:

            n = random.randint(min_redundant_level, max_redundant_level)
            if k < n:
                return k, n
            
    
    def build_bref_sol(max_redundant_level,subsys_num): 
        # structure with: [n, k, selected strategy, subsys_num]
        sol_1 = []
        
        k = k_value[subsys_num] 
        k, n = generate_n_k(min_redundant_level, max_redundant_level, k)
        
        sol_1.append(n)
        sol_1.append(k)
        sol_1.append(random.randint(0,5))#selected strategy
        sol_1.append(subsys_num)
  
        
        return sol_1

    def get_diagonal_np(M):
        return np.diagonal(M).tolist()
    
      
    
    def objetive_function (brief_sol):   
        availability = []   
        #select strategy
        for i in range(num_system):
            if brief_sol[i][2]==0:#(n, working_failure_rate,  repair_rate): 
                CTMC_model = Create_CTMC_model_no_redundancy(brief_sol[i][0],  lamda_w[brief_sol[i][-1]], mu[brief_sol[i][-1]])
            
            elif brief_sol[i][2]==1: # (n, k, working_failure_rate, repair_rate):          
                CTMC_model = Create_CTMC_model_hot_standby(brief_sol[i][0], brief_sol[i][1], lamda_w[brief_sol[i][-1]], mu[brief_sol[i][-1]])

            elif brief_sol[i][2]==2: # (n, k, working_failure_rate, standby_failure_rate, repair_rate, switching_rate):  
                CTMC_model = Create_CTMC_model_warm_standby(brief_sol[i][0], brief_sol[i][1], lamda_w[brief_sol[i][-1]], lamda_s[brief_sol[i][-1]], mu[brief_sol[i][-1]],switch_rate[brief_sol[i][-1]])
            
            elif brief_sol[i][2]==3: # (n, k, working_failure_rate, repair_rate, switching_rate): 
                CTMC_model = Create_CTMC_model_cold_standby(brief_sol[i][0], brief_sol[i][1], lamda_w[brief_sol[i][-1]], mu[brief_sol[i][-1]],switch_rate[brief_sol[i][-1]])
            
            elif brief_sol[i][2]==4: # (n, k, working_failure_rate, standby_failure_rate, repair_rate, switching_rate):
                CTMC_model = Create_CTMC_model_mixed_strategy_active_warm(brief_sol[i][0], brief_sol[i][1], lamda_w[brief_sol[i][-1]], lamda_s[brief_sol[i][-1]], mu[brief_sol[i][-1]],switch_rate[brief_sol[i][-1]])
               
            else:#(n, k, working_failure_rate, repair_rate, switching_rate):
                CTMC_model = Create_CTMC_model_mixed_strategy_active_cold(brief_sol[i][0], brief_sol[i][1], lamda_w[brief_sol[i][-1]], mu[brief_sol[i][-1]], switch_rate[brief_sol[i][-1]])          
                  
            Ergodic_prob = Ergodic_probabilities(CTMC_model)
            
            availability.append(float(calculate_availability(brief_sol[i][0], brief_sol[i][1], Ergodic_prob, brief_sol[i][2])[0]))


        
        A = copy.deepcopy(availability)
        
        U0=	1-	A[0]
        U1=	1-	A[1]
        U2=	1-	A[2]
        U3=	1-	A[3]

     
        
        A_sys = A[0] + U0*A[1]*A[3] +  U0*A[1]*A[2]*U3 # calculating availivbility
        
        cons1 = 0
        cons2 = 0
        for i in range(num_system):
            cons1 = cons1 + brief_sol[i][0]*cost_given[i]
            cons2 = cons2 + brief_sol[i][0]*weight_given[i]

            
        if (cons1-C_limit) > 0  or (cons2-W_limit) > 0:
            A_sys = A_sys*0.0000000000000001 #penalty function
        else:
            A_sys = A_sys

          
        return A_sys, [cons1, cons2]

    
    def Create_CTMC_model_no_redundancy(n, working_failure_rate,  repair_rate): 
    # strategy 0
    
      # Initialize CTMC matrix
    
      CTMC = np.zeros((2,2))
    
      # State no. 0 is subsystem working state
    
      # State no. 1 is subsystem failure state
    
    
    
      # Transition rates
    
      CTMC[0, 1] = n*working_failure_rate
    # Failure of working component
    
      CTMC[1,0] = repair_rate
    # Repair
    
      CTMC[0, 0] = -n*working_failure_rate
    # diagonal
    
      CTMC[1, 1] = -repair_rate
    # diagonal
    
      return CTMC


    def Create_CTMC_model_hot_standby(n, k, working_failure_rate, repair_rate):
    # strategy 1
    
      State_space = np.empty((n-k+2,3), dtype=int)
    
      # States no. from 0 to n-k are subsystem working states
    
      # State no. n-k+1 is subsystem failure state
    
    
    
      # Working states with k working components and remaining in hot standby or failed
    
      j = 0
    
      for i in range(n-k+1):
    
        State_space[i,0] = k 
    # number of working components
    
        State_space[i,1] = n-k-j
    # number of hot standby components
    
        State_space[i,2] = j 
    # number of failed components
    
        j += 1
    
      # Failure states with k-1 operational components and n-k+1 failed components
    
        State_space[n-k+1, 0] = k-1
    # number of working components
    
        State_space[n-k+1, 1] =  0 # number of hot standby components
    
        State_space[n-k+1, 2] = n-k+1
    # number of failed components
    
    
    
      # Initialize CTMC matrix
    
      CTMC = np.zeros((n-k+2, n-k+2))
    
    
      # Transition rates
    
      for i in range(n-k+2):
    
        for j in range(n-k+2):
    
          if State_space[i,0] - State_space[j,0] ==  1: # Failure of operational component
    
           CTMC[i, j] = (State_space[i, 0]+State_space[i, 1]) * working_failure_rate
    
          elif State_space[i, 2] - State_space[j,2] ==  1: # Repair
    
            CTMC[i, j] = State_space[i, 2] * repair_rate
    
    
    
        # Diagonal elements
    
      for i in range(n-k+2):
    
        CTMC[i, i] = -np.sum(CTMC[i, :])
    
    
    
      return CTMC


    def Create_CTMC_model_warm_standby(n, k, working_failure_rate, standby_failure_rate, repair_rate, switching_rate):
    # strategy 2
    
      State_space = np.empty((2*n-2*k+3,3), dtype=int)
    
      # States no. from 0 to n-k are subsystem working states
    
      # States no. form n-k+1 to 2n-2k+2 are subsystem failure states
     
      # Working states with k working components and remaining in warm standby or failed
    
      j = 0
    
      for i in range(n-k+1):
    
        State_space[i, 0] = k 
    # number of working components
    
        State_space[i,1] = n-k-j
    # number of warm standby components
    
        State_space[i,2] = j 
    # number of failed components
    
        j += 1
    
      # Failure states with k-1 operational components and n-k+1 failed components
    
      j = 0
    
      for i in range(n-k+1, 2*n-2*k+3):
    
        State_space[i,0] = k-1
    # number of working components
    
        State_space[i,1] = n-k+1-j
    # number of warm standby components
    
        State_space[i,2] = j 
    # number of failed components
    
        j += 1
    
    
    
      # Initialize CTMC matrix
    
      CTMC = np.zeros((2*n-2*k+3, 2*n-2*k+3))
    
    
    
      # Transition rates
    
      for i in range(2*n-2*k+3):
    
        for j   in range(2*n-2*k+3):
    
          if State_space[i,0] - State_space[j, 0] ==  1 and State_space[j, 2] - State_space[i,  2] ==   1:  # Failure of working component
    
            CTMC[i, j] = State_space[i, 0] * working_failure_rate
    
          elif State_space[i, 1] - State_space[j,  1] ==  1 and State_space[j,  2] - State_space[i, 2] ==  1:  # Failure of warm standby component
    
            if State_space[i, 0] == k - 1:
    
              # CTMC[i, j] = (State_space[i, 1] + State_space[i, 0]) * standby_failure_rate
    
              CTMC[i, j] =  0 # according to the assumption of non-failing components in subsystem failure state
    
            else:
    
              CTMC[i, j] = State_space[i, 1] * standby_failure_rate
    
          elif State_space[i, 1] - State_space[j,  1] ==   1 and State_space[j, 0] - State_space[i, 0] == 1:  # Switch
    
            CTMC[i, j] = switching_rate
    
          elif State_space[i, 2] - State_space[j, 2] ==  1 and State_space[j,1] - State_space[i, 1] ==  1:  # Repair
    
            CTMC[i, j] = State_space[i, 2] * repair_rate
    
    
    
        # Diagonal elements
    
      for i in range(2*n-2*k+3):
    
        CTMC[i, i] = -np.sum(CTMC[i, :])
    
      return CTMC

    def Create_CTMC_model_cold_standby(n, k, working_failure_rate, repair_rate, switching_rate):
    # strategy 3
    
      State_space = np.empty((2*n-2*k+3,3), dtype=int)
    
      # States no. from 0 to n-k are subsystem working states
    
      # States no. form n-k+1 to 2n-2k+2 are subsystem failure states
    
    
    
      # Working states with k working components and remaining in cold standby or failed
    
      j = 0
    
      for i in range(n-k+1):
    
        State_space[i, 0] = k 
    # number of working components
    
        State_space[i, 1] = n-k-j
    # number of cold standby components
    
        State_space[i,2] = j 
    # number of failed components
    
        j += 1
    
      # Failure states with k-1 operational components and n-k+1 failed components
    
      j = 0
    
      for i in range(n-k+1, 2*n-2*k+3):
    
        State_space[i, 0] = k-1
    # number of working components
    
        State_space[i, 1] = n-k+1-j
    # number of cold standby components
    
        State_space[i, 2] = j 
    # number of failed components
    
        j += 1
    
    
    
      # Initialize CTMC matrix
    
      CTMC = np.zeros((2*n-2*k+3, 2*n-2*k+3))
    
    
    
      # Transition rates
    
      for i in range(2*n-2*k+3):
    
        for j  in range(2*n-2*k+3):
    
          if State_space[i, 0] - State_space[j,0] ==  1 and State_space[j,2] - State_space[i, 2] == 1:  # Failure of working component
    
            CTMC[i, j] = State_space[i, 0] * working_failure_rate
    
          elif State_space[i, 1] - State_space[j, 1] ==  1 and State_space[j, 0] - State_space[i, 0] ==  1:  # Switch
    
            CTMC[i, j] = switching_rate
    
          elif State_space[i, 2] - State_space[j, 2] ==  1 and State_space[j, 1] - State_space[i, 1] ==  1:  # Repair
    
            CTMC[i, j] = State_space[i,  2] * repair_rate
    
    
    
        # Diagonal elements
    
      for i in range(2*n-2*k+3):
    
        CTMC[i, i] = -np.sum(CTMC[i,:])
    
    
    
      return CTMC

    
    def Create_CTMC_model_mixed_strategy_active_warm(n, k, working_failure_rate, standby_failure_rate, repair_rate, switching_rate):
    # strategy 4
    
      State_space = np.empty((3*n-3*k+3, 3), dtype=int)
    
    
    
      j = 0
    
      # Working states with k+1 active components
    
      for i in range(n-k):
    
        State_space[i,0] = k+1
    # number of working components
    
        State_space[i, 1] = n-k-1-j
    # number of hot standby components
    
        State_space[i,2] = j 
    # number of failed components
    
        j += 1
    
      j = 0
    
      # Working states with k active components
    
      for i in range(n-k, 2*n-2*k+1):
    
        State_space[i, 0] = k 
    # number of working components
    
        State_space[i, 1] = n-k-j
    # number of hot standby components
    
        State_space[i, 2] = j 
    # number of failed components
    
        j += 1
    
    
    
      j = 0
    
        # Failed states with k-1 active components
    
      for i in range(2*n-2*k+1,3*n-3*k+3):
    
        State_space[i, 0] = k-1
    # number of working components
    
        State_space[i, 1] = n-k+1-j
    # number of hot standby components
    
        State_space[i, 2] = j 
    # number of failed components
    
        j += 1
    
    
    
      # Initialize CTMC matrix
    
      CTMC = np.zeros((3*n-3*k+3, 3*n-3*k+3))
    
    
    
      # Transition rates
    
      for i in range(3*n-3*k+3):
    
        for j  in range(3*n-3*k+3):
    
          if State_space[i,0] - State_space[j,  0] ==  1 and State_space[j,2] - State_space[i, 2] ==     1:  # Failure of active component
    
            CTMC[i, j] = State_space[i, 0] * working_failure_rate
    
          elif State_space[i, 1] - State_space[j, 1] ==  1 and State_space[j,2] - State_space[i, 2] ==  1:  # Failure of warm standby component
    
            if State_space[i, 0] == k - 1:
    
              # CTMC[i, j] = (State_space[i, 1] + State_space[i, 0]) * standby_failure_rate
    
              CTMC[i, j] = 0 # according to the assumption of non-failing components in subsystem failure state
    
            else:
    
              CTMC[i, j] = State_space[i, 1] * standby_failure_rate
    
          elif State_space[i, 1] - State_space[j, 1] ==  1 and State_space[j, 0] - State_space[i, 0] == 1:  # Switch
    
            CTMC[i, j] = (min(min(State_space[i, 1], 2), k +  1 - State_space[i, 0])) * switching_rate
    
          elif State_space[i, 2] - State_space[j, 2] ==  1 and State_space[j, 1] - State_space[i,1] ==  1:  # Repair
    
            CTMC[i, j] = State_space[i, 2] * repair_rate
    
    
    
        # Diagonal elements
    
      for i in range(3*n-3*k+3):
    
        CTMC[i, i] = -np.sum(CTMC[i, :])
    
    
    
      return CTMC

    
    def Create_CTMC_model_mixed_strategy_active_cold(n, k, working_failure_rate, repair_rate, switching_rate):
    # strategy 5
    
      State_space = np.empty((3*n-3*k+3, 3), dtype=int)
    
    
    
      j = 0
    
      # Working states with k+1 active components
    
      for i in range(n-k):
    
        State_space[i,0] = k+1
    # number of working components
    
        State_space[i, 1] = n-k-1-j
    # number of hot standby components
    
        State_space[i, 2] = j 
    # number of failed components
    
        j += 1
    
      j = 0
    
      # Working states with k active components
    
      for i in range(n-k, 2*n-2*k+1):
    
        State_space[i, 0] = k 
    # number of working components
    
        State_space[i, 1] = n-k-j
    # number of hot standby components
    
        State_space[i, 2] = j 
    # number of failed components
    
        j += 1
    
    
    
      j = 0
    
        # Failed states with k-1 active components
    
      for i in range(2*n-2*k+1,3*n-3*k+3):
    
        State_space[i,  0] = k-1
    # number of working components
    
        State_space[i, 1] = n-k+1-j
    # number of hot standby components
    
        State_space[i, 2] = j 
    # number of failed components
    
        j += 1
    
    
    
      # Initialize CTMC matrix
    
      CTMC = np.zeros((3*n-3*k+3,  3*n-3*k+3))
    
    
    
      # Transition rates
    
      for i in range(3*n-3*k+3):
    
        for j   in range(3*n-3*k+3):
    
          if State_space[i,  0] - State_space[j, 0] ==   1 and State_space[j, 2] - State_space[i, 2] ==   1:  # Failure of active component
    
            CTMC[i, j] = State_space[i, 0] * working_failure_rate
    
          elif State_space[i, 1] - State_space[j,  1] ==   1 and State_space[j,  0] - State_space[i,  0] ==   1:  # Switch
    
            CTMC[i, j] =  (min(min(State_space[i,  1],   2), k +  1 - State_space[i, 0])) * switching_rate
    
          elif State_space[i, 2] - State_space[j, 2] ==   1 and State_space[j, 1] - State_space[i,  1] ==  1:  # Repair
    
            CTMC[i, j] = State_space[i, 2] * repair_rate
    
    
    
        # Diagonal elements
    
      for i in range(3*n-3*k+3):
    
        CTMC[i, i] = -np.sum(CTMC[i, :])
    
    
    
      return CTMC

    def Ergodic_probabilities(CTMC):
    
      CTMC_transposed = CTMC.T
    
      CTMC_transposed[0, :CTMC_transposed.shape[0]] =  1
    
      P = np.zeros((CTMC_transposed.shape[0],  1))
    
      P[0,  0] =   1
    
      Ergodic_prob = np.linalg.inv(CTMC_transposed) @ P
    
      return Ergodic_prob
   

    
    def calculate_availability(n, k, Ergodic_prob, selected_strategy):
        #temp_v = 0

        if selected_strategy == 0: # no_redundancy
            temp_v = sum(Ergodic_prob[0:1])
        
        elif selected_strategy == 1:# hot
            temp_v = sum(Ergodic_prob[0:n-k+1])#  States no. from 0 to n-k are subsystem working states

        
        elif selected_strategy == 2:# warm
            temp_v = sum(Ergodic_prob[0:n-k+1])# States no. from 0 to n-k are subsystem working states
   

        elif selected_strategy == 3: #cold
            temp_v = sum(Ergodic_prob[0:n-k+1])# States no. from 0 to n-k are subsystem working states
                 

        elif selected_strategy == 4: # active_warm
            temp_v = sum(Ergodic_prob[0:2*n-2*k+1])# Working states with k+1 active components
 
        else: # active_cold
            temp_v = sum(Ergodic_prob[0:2*n-2*k+1])# Working states with k+1 active components

        Subsystem_availability = temp_v
        
        return Subsystem_availability



    """
    start----------------------------------------
    """  
    
    
    
    ''' Contents used for data storage'''
    The_best_sol_cycle_brief = []
    The_best_sol_cycle = []
    The_availability_cycle = []
    The_cost_cycle = []  
    
    
    availability = []
    cost = []
    
        
        
    max_availability = []
    the_Cost = []
    final_sol = []   
    cycle_result = []
    system_sol_brief = [[] for _ in range(BeeSize)]
    #system_sol = [[] for _ in range(BeeSize)]
    system_sol_brief2 = [[] for _ in range(BeeSize)]
    #system_sol2 = [[] for _ in range(BeeSize)]
    
    system_relibility = [[] for _ in range(BeeSize)]
    system_relibility2 = [[] for _ in range(BeeSize)]
    
    cost_1 = [[] for _ in range(BeeSize)]
    cost_2 = [[] for _ in range(BeeSize)]    

    
    '''======================== initial solution ======================'''
    
    
    for k in range(BeeSize):
        brief_sol =[[] for _ in range(num_system)]
        #availability_sol =[[] for _ in range(num_system)]
        
        for i in range(num_system):           
            brief_sol[i] = build_bref_sol(max_redundant_level,i)    
        
        #for i in range(num_system):
            #availability_sol[i] = creat_sol(brief_sol[i],i)
       
        #print("rand_system1 = ",rand_system1)
        system_sol_brief[k] = brief_sol
        #system_sol[k] = availability_sol
        
    for i in range(BeeSize):
        system_relibility[i], cost_1[i] = objetive_function (system_sol_brief[i])
        #print(system_relibility[i])
    
    localBest_availability = [[] for _ in range(MaxCycle)]
    
    for cycle in range(MaxCycle):

        system_sol_brief2 =  copy.deepcopy(system_sol_brief)
        
        
        '''    
        employed_bees_phase       
        
        '''
       
        for k in range(BeeSize):#[n, k, selected strategy, subsys_num]
            rand = random.randint(1, 3)
            pop_idx = random.choice([index for index in range(BeeSize) if index != k])
            dimension_idx = random.randint(0, num_system - 1)
            if rand == 1: # change a subsystem resource usage
                system_sol_brief2[k][dimension_idx] = system_sol_brief[pop_idx][dimension_idx] 
                
            elif rand == 2: # change a subsystem n  and k usages 
                system_sol_brief2[k][dimension_idx][0] = system_sol_brief[pop_idx][dimension_idx][0] # k
                system_sol_brief2[k][dimension_idx][1] = system_sol_brief[pop_idx][dimension_idx][1] # n
            
            else: # change a subsystem strategy usage
                system_sol_brief2[k][dimension_idx][2] = system_sol_brief[pop_idx][dimension_idx][2] # Strategy
       
        
        
        for i in range(BeeSize):
            system_relibility2[i], cost_2[i] = objetive_function (system_sol_brief2[i])

        for i in range(BeeSize):
            if system_relibility2[i] > system_relibility[i]:
                system_relibility[i] = copy.deepcopy(system_relibility2[i])
                system_sol_brief[i] = copy.deepcopy(system_sol_brief2[i])
                #system_sol[i] =  copy.deepcopy(system_sol2[i])
                num[i] = 0
            else:
                num[i] = num[i] + 1        
        
        
        system_sol_brief2 =  copy.deepcopy(system_sol_brief)

        

        '''      
        onlooker_bees_phase  
          
        '''
        
        prob = []
        prob_N = []
        sum_prob = sum(system_relibility2)
        for i in range(BeeSize):
            prob.append(system_relibility2[i]/sum_prob)
        for i in range(BeeSize):
            prob_N.append((prob[i]-min(prob))/(max(max(prob)-min(prob),1e-30)))
              
        
        
        for k in range(BeeSize):
            if random.random() < prob_N[k]:
                pop_idx = random.choice([index for index in range(BeeSize) if index != k])
                dimension_idx = random.randint(0, num_system - 1)
                rand = random.randint(1, 3)
                if rand == 1:
                    system_sol_brief2[k][dimension_idx] = system_sol_brief[pop_idx][dimension_idx] # 針對特定位置進行改變
 
                elif rand == 2:
                    system_sol_brief2[k][dimension_idx][0] = system_sol_brief[pop_idx][dimension_idx][0] # k
                    system_sol_brief2[k][dimension_idx][1] = system_sol_brief[pop_idx][dimension_idx][1] # n
               
                else:
                    system_sol_brief2[k][dimension_idx][2] = system_sol_brief[pop_idx][dimension_idx][2] # Strategy
                
        
        for i in range(BeeSize):
            system_relibility2[i], cost_2[i] = objetive_function (system_sol_brief2[i])

        for i in range(BeeSize):
            if system_relibility2[i] > system_relibility[i]:
                system_relibility[i] = copy.deepcopy(system_relibility2[i])
                system_sol_brief[i] = copy.deepcopy(system_sol_brief2[i])
                #system_sol[i] =  copy.deepcopy(system_sol2[i])
                num[i] = 0
            else:
                num[i] = num[i] + 1         
        
       
        
        
        """
        scout_bee_phase

        """
        system_sol_brief2 =  copy.deepcopy(system_sol_brief)
     
        
        for i in range(BeeSize):
            
            if num[i] >= Limit:
                for k in range(num_system):           
                    system_sol_brief2[i][k] = build_bref_sol(max_redundant_level,k)    
                    
                system_relibility2[i], cost_2[i] = objetive_function (system_sol_brief2[i])                
                
            num[i] = 0

        for i in range(BeeSize):
            system_relibility2[i], cost_2[i] = objetive_function (system_sol_brief2[i])

        for i in range(BeeSize):
            if system_relibility2[i] > system_relibility[i]:
                system_relibility[i] = copy.deepcopy(system_relibility2[i])
                system_sol_brief[i] = copy.deepcopy(system_sol_brief2[i])
                #system_sol[i] =  copy.deepcopy(system_sol2[i])   
        
        localBest_availability_idx = system_relibility.index(max(system_relibility))
        
        
        if cycle ==0:
            global_brief = system_sol_brief2[localBest_availability_idx]
            #global_sol = system_sol2[localBest_availability_idx]
            global_availability = max(system_relibility)
        
        else:
            if max(system_relibility) > global_availability:
                global_brief = system_sol_brief2[localBest_availability_idx]
                #global_sol = system_sol2[localBest_availability_idx]
                global_availability = max(system_relibility)
        
        
        

        localBest_availability[cycle] = max(system_relibility)
        cycle_result.append(localBest_availability[cycle])
         
        print("globalest_availability_by_cycle = ",global_availability, "cycle = ",cycle, "sim = ", run)
        if cycle == MaxCycle-1:
            global_best.append(max(localBest_availability[cycle],0))
        
        
        


 
        localBest_index = system_relibility.index(max(system_relibility))
        
        The_best_sol_cycle_brief.append(system_sol_brief[localBest_index]) 

        The_availability_cycle.append(max(system_relibility))


    globalBest_index = The_availability_cycle.index(max(The_availability_cycle))
    The_availability_final[run].append(The_availability_cycle[globalBest_index])
    The_final_sol_brief[run].append(The_best_sol_cycle_brief[globalBest_index])
    The_cycle_sol[run].append(The_availability_cycle) 
    The_final_objective_constraints[run].append(objetive_function(The_final_sol_brief[run][0]))

    
    tEnd = time.time()     

    
    '''   
    ---------- output the results -----------------
    
    '''
        
    f = open("4_unit_case.txt", "w")
    
    
    print("The_final_sol_brief", file = f)
    for item in The_final_sol_brief:
        print(str(item), file = f)
    print("\n", file = f)
    
    
    print("The_final_objective_constraints", file = f)
    for item in The_final_objective_constraints:
        print(str(item), file = f)

    print("\n", file = f)
        
    f.close()         
            
       
    print( "It ave_time_cost %f sec" % ((tEnd - tStart)/simulationtime))
    f = open("runtimecost.txt", "w")
    print("%0.10f sec"  % ((tEnd - tStart)/simulationtime), file = f)
        
            
    f.close()        
            
        
        
