import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from ReadData import getPrices #load data
from Technical import bollinger, moving_average_convergence, moving_average, stochastic_oscillator,daily_return,volume #evaluate indicators
from LMS_Algorithm import LMS #defines the approximation function
from IPython.display import clear_output

#initializing the variables

def run_algo(symbol,epochs,initial_train,final_train,initial_test,final_test):
    
    

#    Trains a Q-Learning agent for stock trading
#    Parameters
#    ----------
#    symbol : str
#        Name of the stock on the exchange, ex: APPL, MSFT, PETR4.SA
#    epochs : int
#        Number of epochs for the training
#    initial_train,final_train,initial_test,final_test : str/datetime
#        Training and test period in the format 'AAAA/MM/DD'

#   example of parameters: run_algo('BBAS3.SA',2000,'2013-01-10','2017-01-10','2017-01-10','2017-12-10')
#    Returns
#    ----------
#    new_df : pandas.DataFrame
#        A df like DataFrame with the price column replaced by the log difference in time.
#        The first row will contain NaNs due to first diferentiation.
    
    
    #Initial Portfolio value
    initial_money = 50000
    
    #Definition of colums parameters
    columns = ['Position','NumShares','Money','PortValue','Reward']
    
#############Data Gathering and Organization
    #reading training data from Yahoo
    df_stock = getPrices(symbol,initial_train,final_train)
    
    #Taking rows and columns from Yahoo data
    df_stock.dropna(inplace = True)

    #Creating table to organize daily trades in learning phase
    #Rows = days listed on Yahoo data extracted (daily frequency trading)
    #Columns = Position, #of shares, money, current porfolio value and reward received by the agent
    df_system = pd.DataFrame(index = df_stock.index, columns = columns)
    
##############Reading Data Test
    #reading test data form Yahoo
    df_stock_test = getPrices(symbol,initial_test,final_test)
    
    #Taking rows and columns from Yahoo data
    df_stock_test.dropna(inplace = True)
    
    #Creating table to organize daily trades in learning phase
    #Rows = days listed on Yahoo data extracted (daily frequency trading)
    #Columns = Position, #of shares, money, current porfolio value and reward received by the agent
    df_system_test = pd.DataFrame(index = df_stock_test.index, columns = columns)
    
    #Total number of operations the agent can perform = number of days - 1
    total_periods = np.size(df_stock,axis = 0) - 1
    
##############Setting configuration learn table
    #Set total portfolio value as avaliable from day 1
    df_system.loc[0:1,'Money'] = initial_money
    
    #Replace all non existing elements with zeros
    df_system = df_system.fillna(0)
    
    #Create close price column
    df_system['Close'] = df_stock['Close']
    
##############Setting configuration test table
    #Same as above
    df_system_test.loc[0:1,'Money'] = initial_money
    df_system_test = df_system_test.fillna(0)
    df_system_test['Close'] = df_stock_test['Close']
    
##############Setting States Vector
    #Defining 4 possible states
    num_states = 4
    
    #Defining function that returns the state of the agent
    def getStates(df,num_states):
        
        #States variable = array filled with zeros, size 3 x 4
        states = np.zeros([np.size(df,axis = 0),num_states])
        
        #Second column of states variable = daily return
        states[:,1] = daily_return(df)
        
        #First column of states variable = bollinger
        states[:,0] = bollinger(df,20)[0]
        
        #third column of states variable = moving average
        states[:,2] = moving_average_convergence(df)[2]
        
        #Last (4th) column of states variable = stochastic
        states[:,3] = (stochastic_oscillator(df)[2] -0.5)
        
        #check for Nan elements in states matrix and replace them with zeros
        states[np.isnan(states)] = 0

        return states
    
    #Assignes new states variable for learning phase and test phase as a result of state function above
    states = getStates(df_stock,num_states)
    states_test = getStates(df_stock_test,num_states)
    
    #Defining function for portoflio value update
    def update(df):
        #Total portoflio value = portoflio value + money generated from daytrade
        df['Total'] = df['Money'] + df['PortValue']
        
    #Updates learn and test tables portoflio value from function above
    update(df_system)
    update(df_system_test)
    
##############Initializing LMS algorithm with random weights with correct dimension
    #number of parameters (input dimension) = number of states = 4
    n = num_states 
    
    #number of possible actions (output dimensions) = buy, sell or hold
    m = 3 
    
    #defining random weights for LMS model, matriz 3 x 4
    w_initial = np.squeeze(np.random.rand(m,n)/100)
    
    #initialize LMS algorithm with random weights with the correct dimensions
    model = LMS(0.05,w_initial)
    
##############Defining reward function for agent
    def getReward(i,df,state):
        
        #Reward is a function of the portoflio value variation and the stock value variation
        #R = 1,5 x [((Pn - Pn-1)/Pn-1) - ((Sn - Sn-1)/Sn-1)] (!)
        reward = 100*(1.5*((df.iloc[i]['Total'] - df.iloc[i-1]['Total'])/df.iloc[i-1]['Total']) - (df.iloc[i]['Close'] - df.iloc[i-1]['Close'])/df.iloc[i-1]['Close'])
        return reward
    
    # (!)
    df_system.loc[0:1,'Money'] = initial_money
    df_system_test.loc[20:21,'Money'] = initial_money
    
##############Setting time
    #importing time function
    import time
    
    #set start time to count learning period duration
    start_time = time.time()
        
##############Setting Learning parameters
    #epsilon = exploration rate
    #we should have a high epislon at the beggining of the trainning as reduce it progressively as
    #the agent becomes more confident at estimating Q-values
    epsilon = 1
    
    #gamma = discount factor
    #discounts value of future rewards, introducing time as a variable in the decision making
    gamma = 0.95
    
    #Y is used to assess whether the convergence of Q was achieved or not 
    Y = np.zeros((epochs,3))
    cout = 0
    reco = np.zeros(epochs)
    rewa = np.zeros(epochs)
  
##############Learning loop   
    for k in range(epochs):
        
        #Initializing counter
        i = 1
        
        #Defining state from counter set
        state = states[i]
        
        #Define position
        position = 0
        
        #Initializing loop in time series
        while (total_periods > i):
            
            #Aproximating Q value
            qval = model.predict(state.reshape(1,num_states))
            
            #Define action of exploration or explotation from random variable
            #Random.randon() between 0 - 1
            #If random variable is less than epsilon, define action: -1, 0, 1, 2
            if (random.random() < epsilon): #explore x exploit
                action = np.random.randint(-1,2)
            
            #In random variable is greater than epsilon, define action as the maximum value of Q array - 1 (!)
            else:
                action = np.argmax(qval)-1
            
            #Track action taken in each day
            df_system.ix[i,'Position'] = action
            
            #Action listing
            #If long: 
            #Action -1: Maintains position
            #Action 0: Maintains position
            #Action 1: Ends position
            #Action 2: Doubles position (!) 
            #If short:
            #Action -1: Ends position
            #Action 0: Maintains position
            #Action 1: Inverts position? (!)
            #Action 2: Doubles and inverts position? (!)
            #If out:
            #Action -1: Shorts
            #Action 0: Maintains
            #Action 1: Long
            #Action 2: Double long? (!)
            if action == 1 and df_system.ix[i-1,'NumShares'] > 0 :
                numshare = 0 #numshare is a variable that tells us how many shares are bought/sold in this state
            elif action == -1 and df_system.ix[i-1,'NumShares'] < 0:
                numshare = 0
            elif df_system.iloc[i-1]['NumShares'] !=0 :
                numshare = np.abs(df_system.iloc[i-1]['NumShares']) * df_system.iloc[i]['Position']
            else:
                numshare = np.int(df_system.iloc[i-1]['Total']/df_system.iloc[i]['Close']) * df_system.iloc[i]['Position']
            
            #updating the dataframe
            df_system.ix[i,'Money'] = df_system.iloc[i-1]['Money'] - numshare*df_system.iloc[i]['Close']
            df_system.ix[i,'NumShares'] = df_system.iloc[i-1]['NumShares'] + numshare         
            df_system.ix[i,'PortValue'] = df_system.iloc[i]['NumShares']*df_system.iloc[i]['Close']     
            df_system['Total'] = df_system['Money'] + df_system['PortValue']
            
            #Goes to next state
            new_state = states[i+1]
            
            #Evaluate reward
            reward = getReward(i,df_system,position)
            
            #Attributes reward
            df_system.ix[i,'Reward'] = reward
            
            #Aproximates Q and gets maximum value
            newQ = model.predict(new_state.reshape(1,num_states))
            maxQ = np.max(newQ)
            
            #Defines zero matrix of 1 line and  columns
            y = np.zeros((1,3))
            y[:] = qval[:]
            
            #Updates Q-Learning value
            update = (reward + (gamma * maxQ))
            
            #updates only the Q of the action taken
            y[0][action+1] = update    
            
            #weight updates
            model.fit(state.reshape(1,num_states), y) 
            state = new_state
            
            #Increments counter
            i+=1
            
            #Clear output
            clear_output(wait=True)
            if epsilon > 0.1:  #reduces epsilon
                epsilon -= (0.5/epochs)
            if (epochs - k) < 10: # epsilon goes to zero in the last 10 epochs
                epsilon = 0
            
        reco[cout] = (df_system.iloc[-2]['Total'] - df_system.iloc[2]['Total'])/df_system.iloc[2]['Total']
        Y[cout] = y  # Y is a array with Q values of the last time period, we can evaluate the convergence of Q using this
        rewa[cout] = np.sum(df_system['Reward'])
        cout+=1
    
    #Get summary of learning period    
    summary = df_system[['Position','NumShares','Reward','Close','Total']] 
    
    #benchmark for learning period
    bench = (df_system.iloc[-2]['Close'] - df_system.iloc[2]['Close'])/df_system.iloc[2]['Close']
    
    #print learning period time
    print("--- %s seconds ---" % (time.time() - start_time))

    #Defines function for algorithm performance test
    def test_algo(XD,states,df):
        
        #Get dataframe
        df_test_algo = df.copy()   

        #Number of iterations       
        total_periods = np.size(states,0)
        
        #initiate counter
        i = 1
        while(total_periods > i):
            
            #PQ value as product of state and XD.T (!)
            Q = np.dot(states[i],XD.T)
            
            #Define action as function of maximum Q value
            action = np.argmax(Q)-1
            
            #Map action taken in matrix
            df_test_algo.ix[i,'Position'] = action
            
            #Take action
            if action == 1 and df_test_algo.ix[i-1,'NumShares'] > 0 :
                numshare = 0 #numshare is a variable that tells us how many shares are bought/sold in this state
            elif action == -1 and df_test_algo.ix[i-1,'NumShares'] < 0:
                numshare = 0
            elif df_test_algo.iloc[i-1]['NumShares'] !=0 :
                numshare = np.abs(df_test_algo.iloc[i-1]['NumShares']) * df_test_algo.iloc[i]['Position']
            else:
                numshare = np.int(df_test_algo.iloc[i-1]['Total']/df_test_algo.iloc[i]['Close']) * df_test_algo.iloc[i]['Position']
            
            #Update dataframe
            df_test_algo.ix[i,'Money'] = df_test_algo.iloc[i-1]['Money'] - numshare*df_test_algo.iloc[i]['Close']
            df_test_algo.ix[i,'NumShares'] = df_test_algo.iloc[i-1]['NumShares'] + numshare         
            df_test_algo.ix[i,'PortValue'] = df_test_algo.iloc[i]['NumShares']*df_test_algo.iloc[i]['Close']     
            df_test_algo['Total'] = df_test_algo['Money'] + df_test_algo['PortValue']
            i+=1
        return df_test_algo
        
    #Run algo using model, states matrix and dataframe test matrix
    X = test_algo(model.w,states_test[20:],df_system_test[20:])
    
    #Print developed model
    print model.w
    
    #Result of algorithm performance and stock performance over the period
    result = (X.iloc[-2]['Total'] - X.iloc[2]['Total'])/X.iloc[2]['Total']
    bench_test = (X.iloc[-2]['Close'] - X.iloc[2]['Close'])/X.iloc[2]['Close']
    #print '\n'
    #print 'Inicio do teste %s, fim do teste %s'  %(initial_test,final_test)
    #print 'Resultado teste %f' %result
    #print 'Benchmark teste %f' %bench_test  
    #print 'Inicio do treino %s, fim do treino %s'  %(initial_train,final_train)
    #print 'Resultado treino %f' %reco[-1]
    #print 'Benchmark treino %f' %bench 
    
    return result
    
    #Clear
    df_system = 0
    df_system_test = 0
