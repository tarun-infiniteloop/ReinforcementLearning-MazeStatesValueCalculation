
import numpy as np
import math

class Maze:
    def __init__(self, gridHeight=6, gridWidth=6, terminalReward=10, lockPickProb=0.5):
        self.rewardsLeft = np.array([[-1, 0, 0, 0, 0, 0], 
                                    [-1, -1, 0, 0, 0,-10], 
                                    [-1, 0, 0, -1, -1, -1], 
                                    [0, 0, 0, -10, -1, -1],
                                    [-1, -1, 0, 0, -1, 0],
                                    [-1, 0, -1, 0, 0 ,-1]])

        self.rewardsRight =  np.array([[ 0, 0, 0, 0, 0, -1], 
                            [ -1, 0, 0 , 0, -10, -1],
                            [ 0, 0, -1, -1, -1, -1],
                            [ 0, 0, -10, -1, -1 ,-1],
                            [ -1, 0, 0, -1, 0, -1],
                            [ 0, -1, 0, 0, -1, -1]])

        self.rewardsUp  =  np.array([[ -1, -1, -1, -1, -1, -1], 
                            [ 0, -1, -1, -1, -1, 0],
                            [ 0, 0, -1, 0, 0, 0],
                            [ -1, 0,0, 0,0, 0],
                            [ 0, -10, -1, -1, -1, 0],
                            [ 0,  0, -1, -10, 0, 0]])


        self.rewardsDown =  np.array([[ 0, -1, -1, -1, -1, 0], 
                            [ 0, 0, -1, 0, 0, 0],
                            [ -1, 0, 0, 0, 0, 0],
                            [ 0, -10,-1,-1,-1, 0],
                            [  0,0,-1,-10,0, 0],
                            [ -1, -1, -1, 0, -1, -1]])

        self.gridHeight = gridHeight
        self.gridWidth = gridWidth
        self.lockPickProb = lockPickProb
        self.terminalReward = terminalReward


    def isStateTerminal(self, state):
        if state == (3, 0) :
            return True
        elif state == (5, 3):
            return True
        return False

    def takeAction(self, state, action):
        retVal = []
        if(self.isStateTerminal(state)):
            return [[state,1, self.terminalReward]] 

        if action=='left':
            reward = self.rewardsLeft[state]
            if(reward == -1):
                retVal.append([state,1,-1])
            elif(reward == -10):
                retVal.append([(state[0], state[1]-1),self.lockPickProb,-1])
                retVal.append([state,1-self.lockPickProb,-1])
            else:
                retVal.append([(state[0], state[1]-1),1,-1])

        if action=='right':
            reward = self.rewardsRight[state]
            if(reward == -1):
                retVal.append([state,1,-1])
            elif(reward == -10):
                retVal.append([(state[0], state[1]+1),self.lockPickProb,-1])
                retVal.append([state,1-self.lockPickProb,-1])
            else:
                retVal.append([(state[0], state[1]+1),1,-1])

        if action=='up':
            reward = self.rewardsUp[state]
            if(reward == -1):
                retVal.append([state,1,-1])
            elif(reward == -10):
                retVal.append([(state[0]-1, state[1]),self.lockPickProb,-1])
                retVal.append([state,1-self.lockPickProb,-1])
            else:
                retVal.append([(state[0]-1, state[1]),1,-1])

        if action=='down':
            reward = self.rewardsDown[state]
            if(reward == -1):
                retVal.append([state,1,-1])
            elif(reward == -10):
                retVal.append([(state[0]+1, state[1]),self.lockPickProb,-1])
                retVal.append([state,1-self.lockPickProb,-1])
            else:
                retVal.append([(state[0]+1, state[1]),1,-1])
        for i,[nextState, prob, reward] in enumerate(retVal):
            if(self.isStateTerminal(nextState)):
                retVal[i][2] = self.terminalReward   

        return retVal 

class GridworldSolution:
    def __init__(self, maze,horizonLength):
        self.env = maze
        self.actionSpace = ['left', 'right', 'up',  'down']
        self.horizonLength = horizonLength
        self.DP = np.ones((self.horizonLength + 1, self.env.gridHeight, self.env.gridWidth), dtype = float) * -np.inf
    
    def optimalReward(self, state, k):
        optReward = -np.inf
        
        #### Write your code here
        # set values at HORIZON time step
        
        #set value "zero" because at Horizon time step there will be no action, so there will be no reward in any state            
        self.DP[self.horizonLength] = 0
        
        for timeInstance in range(self.horizonLength - 1, k-1, -1):
            #for each state in present timeInstance, check for which action expected reward is maximum   
            # futureMatrix is used to get cost of next state after taking action which we get from future matrix
            futureMatrix = self.DP[timeInstance + 1]
            for x in range(self.env.gridHeight):
                for y in range(self.env.gridWidth):
                    #for each action check maximum expected reward and then finally update that reward in (x,y) state
                    maxExpReward = -np.inf
                    for action in self.actionSpace:
                        #traCostList - it is a single stage cost or transition cost
                        traCostList = self.env.takeAction((x,y), action)
                        tempReward = -np.inf
                        if len(traCostList) == 2:
                            tempReward = traCostList[0][1]*(traCostList[0][2] + futureMatrix[traCostList[0][0][0]][traCostList[0][0][1]]) + traCostList[1][1]*(traCostList[1][2] + futureMatrix[traCostList[1][0][0]][traCostList[1][0][1]])
                        else : 
                            tempReward = traCostList[0][2] + futureMatrix[traCostList[0][0][0]][traCostList[0][0][1]] 
                        #update maxExpReward by comparision of given action with other actions took place 
                        if tempReward > maxExpReward:
                            maxExpReward = tempReward
                    #update (x,y) state maximumExpectedReward
                    self.DP[timeInstance][x][y] = maxExpReward
        
        optReward = self.DP[k][state[0]][state[1]]  

        ########
        return optReward

if __name__ == "__main__":
    maze = Maze()
    solution = GridworldSolution(maze,horizonLength=5)
    print(" Horizon ",solution.horizonLength)
    optReward = solution.optimalReward((2,0),0)
    print(optReward)
