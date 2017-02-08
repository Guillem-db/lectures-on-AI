import gym
import numpy as np
import copy
import time 
import pandas as pd
from gym.monitoring import Monitor
from collections import Counter
from IPython.core.display import clear_output
class TaxiModel(object):
    

    
    def __init__(self,seed=0,
                 win_cond=None,
                 dead_cond=None,
                 score=None,
                 distance=None,
                 peste=0.95,
                 gamma=0,
                ):
        self.gama=gamma
        self._distance = distance
        self._score =score
        self._win_cond = win_cond
        self._dead_cond = dead_cond
        self.mdp = self.get_mdp()
        self.env = gym.make('Taxi-v1')
        self.data = self.init_data()
        self.monitor = None
        self.seed = seed
        self.obs = {}
        s0 = self.env.reset()
        self._state = int(s0)
        self.penalty = 0
        self.end = False
        self.reads = 0
        self.peste = peste
        self.policy = []
        self.sample_policy = {}
        print(s0)
    
    def potential(self,state):
        state = list(self.decode(state))
        taxy = np.array((state[0],state[1]))
        if state[2]< 4:
            pas = np.array(self.env.locs[state[2]])
        des = np.array(self.env.locs[state[3]])
        if state[2] == 4:
            return np.linalg.norm(taxy-des)+1 
        else:
            return np.linalg.norm(taxy-pas)+1 +100
        
    def get_value_function(self,n_iters=1000):
        n_states = 500
        n_actions = 6
        value_function = np.zeros(n_states)
        def get_action_value(mdp, s, a, value_function):
            value = 0
            x = mdp[s][a][0]
            value = value_function[x[1]] + x[2]
            #print(x,value)
            return value
        for i in range(n_iters):
            #clear_output(wait=True)
            #print(i)
            for s in range(n_states):
                new_value = 0
                for a in range(n_actions):
                    ac_val = get_action_value(self.mdp, s, a, value_function)
                    new_value = max(new_value, ac_val)
                    #if s==0:
                    #print("iter: ",i, "state: ",s, "action: ",a," new_value: ",new_value,"acc val", ac_val, "value function: ",value_function[s],
                    #          "mdo",mdp[s][a][0],"x[0]",mdp[s][a][0][0],"x1",mdp[s][a][0][2])
                value_function[s] = new_value
        return value_function
    
    @property
    def seed(self):
        return self._seed[:]
    @seed.setter
    def seed(self,seed):
        self._seed = self.env.seed(seed)
        np.random.seed(self._seed)
    
    @property
    def state(self):
        return self._state
    @staticmethod
    def relativize(x):
            if x<=0:
                return np.exp(x)
            if x>0:
                return 1+np.log(1+x)
    def init_data(self):
        def alt_dead_cond(ix,row):
            return ix[1]<4 and ix[0]==row['newstate']
            
        mdp_dict = self.mdp
        taxi = pd.DataFrame.from_dict(mdp_dict)
        taxi = taxi.unstack()
        vals = np.array([x[0] for x in taxi.values])
        cols = ['one','newstate','reward','done']
        df = pd.DataFrame(index= taxi.index,columns=cols,data=vals)
        df.index.levels[0].name = 'State'
        df.index.levels[1].name = 'Action'
        df['win'] = False
        df.loc[df['reward']==20.,'win'] = True
        df['dead'] = False
        df.loc[df['reward'].values==-10.,'dead'] = True
        df.loc[:,'newstate'] = df.newstate.values.astype(int)
        df['row'] = np.nan
        df['col'] = np.nan
        df['pass'] = np.nan
        df['dest'] = np.nan
        df['N'] = 0
        df['this_N'] = 0
        df['prob'] = 1./6.
        df.loc[:,['row','col','pass','dest']] = [list(self.decode(ix[0])) for ix,row in df.iterrows()]
        df.loc[:,'desc'] = [str(list(self.decode(ix[0]))) for ix,row in df.iterrows()]
        val_func = self.get_value_function()
        df['value_func'] = [val_func[ix[0]] for ix in df.index]
    
        df['dist'] = [self.potential(ix[0]) for ix in df.index]
        df['inv_dist'] = -df['dist'].values
        df['score'] = df['reward'].map(self.relativize).values*1./df['dist'].values
        
        alt_dead = np.array([alt_dead_cond(ix,row) for ix,row in df.iterrows()])
        df.loc[:,'dead'] = df['dead'].values | alt_dead
        cond1 = df['pass'].values== 4
        cond2 = df['dead'].values== False
        df.loc[cond1 & cond2,'reward'] += 10
        df.loc[alt_dead,'reward'] -= 10
        descs = df[df['win']==True]['desc'].values
        cond = lambda x: x['desc'] in descs
        ix = [cond(r) for ix,r in df.iterrows()]
        
        df['can_win'] = False
        df.loc[ix,'can_win'] = True
        old = df.index
        df= df.reset_index()
        df.index=old
        df['peste'] =np.ones(len(df.index))
        num_dead=df['dead'].groupby(level=0).apply(lambda x: sum([int(i) for i in x]))
        df['least_action'] = -1
        df['best_former'] = -1
        df['best_reward'] = -1
        df['fut_score'] = 0
        df['former_state'] = -1
        df['dist_end'] = 10e8
        df['policy'] = np.nan
        for ix, row in df.iterrows():
            df.loc[ix[0],'dead_states'] = num_dead[ix[0]]
        return df
    
    def clear_data(self):
        df = self.data.copy()
        df['least_action'] = -1
        df['best_former'] = -1
        df['best_reward'] = -1
        df['fut_score'] = 0
        df['former_state'] = -1
        df['dist_end'] = 10e8
        df['policy'] = np.nan
        df['this_N'] = 0
        df.loc[ df['peste'] >0,'peste']=1
        df['prob'] = 1./6.
        self.data = df.copy()
        
    def update_memory(self,state, action):
    
        visited = self.data[self.data['this_N']>0]
        visited
        state_reward = self.data.ix[(state,action),'best_reward']#independent of action
        next_state = self.data.ix[(state,action),'newstate']
        next_state_reward = self.data.ix[(next_state,action),'best_reward'] 
        no_loop = self.data.ix[(state,action),'best_former']!=next_state
        improves_reward = next_state_reward == -1 or state_reward+1 < next_state_reward
        dead = self.data.ix[(state,action),'dead']
        if improves_reward and no_loop:# and next_state != state and not dead:
            self.data.loc[next_state,'best_reward'] = state_reward + 1
            self.data.loc[next_state,'best_former'] = state
            self.data.loc[next_state,'least_action'] = action
            self.data.loc[(state,action),'peste'] = 2
            if state_reward+1 < next_state_reward:
                self.filter_memory(next_state)
            visited = self.data[self.data['this_N']>0]
            
                
    def filter_memory(self,state):
        visited = self.data[self.data['this_N']>0]
        cond = visited['best_former'].values == state
        states = visited.ix[cond,'State'].copy()
        if len(states) == 0:
                return
        state_reward = int(self.data.ix[(state,0),'best_reward'])
        for next_state in states.values:
            #print("filtering:",next_state)
            self.data.loc[next_state,'best_reward'] = state_reward + 1
            if visited.ix[next_state,'can_win'].any():
                return
            else:
                self.filter_memory(next_state)
        
            
        
    @staticmethod
    def encode(taxirow, taxicol, passloc, destidx):
        i = taxirow
        i *= 5
        i += taxicol
        i *= 5
        i += passloc
        i *= 4
        i += destidx
        return i

    @staticmethod
    def decode(i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def sample(self,state=0,size=1):
        if bool(np.random.binomial(1,self.gamma)) and state in self.sample_policy.keys():
            return np.array([self.sample_policy[state] for _ in range(size)])
        return np.array([self.env.action_space.sample() for _ in range(size)])
    
    
    def sample_memory(self,state,size=1):
        
        if bool(np.random.binomial(1,self.gamma)) and state in self.sample_policy.keys():
            return np.array([self.sample_policy[state] for _ in range(size)])
        pest = self.data.ix[state,'peste'].values
        probs = pest#np.ones(len(pest))
        probs[pest==0] = 0
        probs = probs/probs.sum()
        return np.random.choice(np.arange(6),p=probs,size=(size))
        #return np.array([self.env.action_space.sample() for _ in range(size)])
    
    def get_mdp(self):
        self.MAP = [
            "+---------+",
            "|R: | : :G|",
            "| : : : : |",
            "| : : : : |",
            "| | : | : |",
            "|Y| : |B: |",
            "+---------+",
        ]
        desc = np.asarray(self.MAP,dtype='c')
        locs = locs = [(0,0), (0,4), (4,0), (4,3)]

        nS = 500
        nR = 5
        nC = 5
        maxR = nR-1
        maxC = nC-1
        isd = np.zeros(nS)
        nA = 6
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        for row in range(5):
            for col in range(5):
                for passidx in range(5):
                    for destidx in range(4):
                        if passidx < 4 and passidx != destidx:
                            isd[state] += 1
                        for a in range(nA):
                            state = self.encode(row, col, passidx, destidx)
                            # defaults
                            newrow, newcol, newpassidx = row, col, passidx
                            reward = -1
                            done = False
                            taxiloc = (row, col)

                            if a==0:
                                newrow = min(row+1, maxR)
                            elif a==1:
                                newrow = max(row-1, 0)
                            if a==2 and desc[1+row,2*col+2]==b":":
                                newcol = min(col+1, maxC)
                            elif a==3 and desc[1+row,2*col]==b":":
                                newcol = max(col-1, 0)
                            elif a==4: # pickup
                                if (passidx < 4 and taxiloc == locs[passidx]):
                                    newpassidx = 4
                                else:
                                    reward = -10
                            elif a==5: # dropoff
                                if (taxiloc == locs[destidx]) and passidx==4:
                                    done = True
                                    reward = 20
                                elif (taxiloc in locs) and passidx==4:
                                    newpassidx = locs.index(taxiloc)
                                else:
                                    reward = -10
                            newstate = self.encode(newrow, newcol, newpassidx, destidx)
                            P[state][a].append((1.0, newstate, reward, done))
        return P
    
    
    
    def score(self,state,dec,penalty):
        if self._score is None:
            if self.win_cond(state,dec):
                print("win found: ",penalty)
            elif self.data.ix[(state,dec),'pass']==4:
                print("PASS PICKED: ",penalty)
            return float(self.data.ix[(state,dec),'score'])#-penalty
        else:
            return self._score(self,state,dec,penalty)
    def win_cond(self,state, dec):
        if self._win_cond is None:
            return bool(self.data.ix[(state,dec),'win'])
        else:
            return self._win_cond(self,state,dec)
    def dead_cond(self,state, dec):
        if self._dead_cond is None:
            return bool(self.data.ix[(state,dec),'dead'])
        else:
            return self._dead_cond(self,state,dec)
    
    def tick_step(self,state,dec):
        #old_state = int(self.env.s)
        self.data.loc[(state,dec),'peste'] = self.data.ix[(state,dec),'peste'] * self.peste
        if self.data.loc[(state,dec),'dead']:
            self.data.loc[(state,dec),'peste'] = 0.
        self.data.loc[(state,dec),'this_N'] = self.data.ix[(state,dec),'this_N'] + 1
        self.data.loc[(state,dec),'N'] = self.data.ix[(state,dec),'N'] + 1
        self.update_memory(state,dec)
        self.reads += 1
        return int(self.data.ix[(state,dec),'newstate'])
    
    
    def distance(self,a,b,a_dec,b_dec):
        if self._distance is None:
            arr0 = self.data.ix[(a,0),['row','col','pass','dest']].values.tolist()
            arr0[2] = [0 if arr0[2]<4 else 10][0]
            if a_dec in [0,1,2,3] and a_dec != b_dec:
                a_mom = 10
            else:
                a_mom = 0
            x0 = np.array(arr0[:3]+[a_mom])
            arr1 = self.data.ix[(b,0),['row','col','pass','dest']].values.tolist()
            arr1[2] = [0 if arr1[2]<4 else 10][0]
            x1 = np.array(arr1[:3]+[0])
            return np.linalg.norm(x0-x1)**2
        else:
            return self._distance(self,a,b,a_dec,b_dec)
    
    def step(self, decision):
        obs = self.env.step(decision)
        self._state = self.env.s
        self.obs['pos'] = obs[0]
        self.obs['reward'] = obs[1]
        self.obs['end'] = obs[2]
        self.obs['info'] = obs[3]
        if not self.monitor is None:
            self.env.render()
        return obs
    
    def start_monitor(self,outdir='/home/kalidus/code/openai'):
        self.monitor = Monitor(self.env)
        self.monitor.start(outdir,force=True)
        
    def _propagate_score(self,state,max_score,dist_end):
        best_former = self.data.ix[(state,0),'best_former']
        #print("state:",state,"has best former:",best_former)
        least_action =  self.data.ix[(state,0),'least_action']
        repe = self.policy[-1] == (best_former,least_action)
        if not state==self._state and not repe:
            self.policy.append((best_former,least_action))
            return
        self.data.loc[(best_former,least_action),'fut_score'] = int(max_score)
        self.data.loc[(best_former,least_action),'dist_end'] = int(dist_end)
        for action in range(6):
            if self.data.loc[(best_former,action),'peste'] != 0:
                self.data.loc[(best_former,action),'peste'] = 1
        self.data.loc[(best_former,least_action),'peste'] = 10
   
    def propagate_policy(self):
        self.data.loc[:,'fut_score'] = 0
        self.data.loc[:,'peste'] = self.data.ix[:,'peste'] * 1.5
        active = self.data['this_N'] > 0
        max_score = self.data.ix[active,'score'].max()
        best_index = self.data.ix[active,'score'] == max_score
        first_best  =  self.data.ix[active & best_index,'State'].values[0]
        if not self.data.ix[(first_best,0),'can_win']:
            return False
        #print("best_state:",best_state,"max_score:",max_score)
        dist_end = 1
        active = self.data['this_N'] > 0
        cont_cond = (self.data.ix[active,'dist_end'] == 10e8).any()
        self.policy = [(first_best,5)]
        best_state = int(first_best)
        i = 0
        while best_state!=self._state and i<=1000:#self.data.ix[(first_best,0),'best_reward']:
            #print(best_state)
            self._propagate_score(best_state,max_score,dist_end)
            dist_end += 1
            best_state = int(self.data.ix[(best_state,0),'best_former'])
            
            i+=1
        """best_former = self.data.ix[(best_state,0),'best_former']
        least_action =  self.data.ix[(best_state,0),'least_action']
        self.policy.append((best_former,least_action))"""

        if best_state == self._state:
            self.policy = list(reversed(self.policy))
            for st,ac in self.policy:
                self.data.loc[st,'policy'] = ac
            return True
        else:
            return False
            
    def make_decision(self):
        state = int(self.env.s)
        mp = self.data.ix[state,'policy']
        self.data[self.data['policy']==mp]['Action']
        actions = self.data[self.data['policy']==mp]['Action']
        if len(actions) == 1:
            return int(actions)
        else:
            return np.random.choice(actions.values)