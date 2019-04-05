import numpy as np
import random
import math

class Game:
    old_state = None
    old_action = None
    
    
    def __init__(self,h,w,par1=0,par2=0,par3=0,th=0,tw=0):
        self.e = 0.01
        self.alpha = 0.1
        self.gamma = 0.9
        self.rate = 0.99
        self.rewmove = -0.1
        self.rwd = 10
        self.lr = 0.05
        self.thres = 0.5
        self.h = h
        self.w = w
        self.a = np.zeros((h,w))
        self.a[0]=1
        self.a[-1]=1
        self.a[:,0]=1
        self.a[:,-1]=1
        self.posx=0
        self.posy=0
        self.th = th
        self.tw = tw
        self.par1 = par1
        self.par2 = par2
        self.par3 = par3
        self.reward = np.ones((self.h,self.w,4))*self.rewmove
        self.transition = np.zeros((self.h,self.w,5))
                
        
    def reset(self):
        self.a = np.zeros((self.h,self.w))
        self.a[0]=1
        self.a[-1]=1
        self.a[:,0]=1
        self.a[:,-1]=1
        self.stenka(self.par1,self.par2,self.par3)
        self.posy = random.randint(1, self.par1-1)
        self.posx = random.randint(1, self.h-2)
        self.a[self.posx,self.posy]=5
        self.a[self.th,self.tw]=8
        
    def draw(self):
        print(self.a)
  
    
    def stenka(self,par1,par2,par3):
        self.par1=par1
        self.par2=par2
        self.par3=par3
        if par1<2 or par1>self.w+2:
            print('wrong parametr par1')
            return
        if par2>par3:
            print('wrong parametr par2')
            return
        if par3>self.h:
            print('wrong parametr par3')
            return
        self.a[:,par1]=1
        for i in range(par3-par2):
            self.a[par2+i,par1]=0
            
        for i in range(self.h):
            for j in range(self.w):
                if self.a[i,j]!=1:
                    if self.a[i-1,j]!=1:
                        self.transition[i,j,0]=1
                    if self.a[i+1,j]!=1:
                        self.transition[i,j,1]=1
                    if self.a[i,j-1]!=1:
                        self.transition[i,j,2]=1
                    if self.a[i,j+1]!=1:
                        self.transition[i,j,3]=1     
    
    
    def update(self,xold,yold,xnew,ynew):
        if xnew == self.th and ynew == self.tw:
            print('GAME WIN')
          #  self.__init__(self.h,self.w,par1=self.par1,par2=self.par2,par3=self.par3,th=self.th,tw=self.tw)
          #  self.stenka(self.par1,self.par2,self.par3)
          #  self.tcel(self.th,self.tw)
          #  self.draw()   
            return 'win'
        k = self.a[xold,yold]
        self.a[xold,yold] = self.a[xnew,ynew]
        self.a[xnew,ynew] = k
        return 'move'
      
    def play(self):
        k=True
        while k:
            key = ord(getch())
            print(key)
            if key==27:
                k=False
            if key == 80: #Down arrow
                self.down()
            elif key == 72: #Up arrow
                self.up()
            elif key == 75: #Left arrow
                self.left()
            elif key == 77: #Right arrow
                self.right()
            print(self.a)    
    
    def tcel(self,h,w):
        self.th=h
        self.tw=w
        self.a[h,w]=8
        
        trans = self.transition[h,w]
        if trans[0]==1:
            self.reward[self.th-1,self.tw,1]=self.rwd
        if trans[1]==1:
            self.reward[self.th+1,self.tw,0]=self.rwd
        if trans[2]==1:
            self.reward[self.th,self.tw-1,3]=self.rwd
        if trans[3]==1:
            self.reward[self.th,self.tw+1,2]=self.rwd
        
    def up(self):
        xold,yold = self.posx,self.posy
        if self.a[self.posx-1,self.posy] ==1:
            print('udar v stenku!')
            a,tr = 'udar',[self.posx,self.posy]
        else:
            self.posx = self.posx-1
            a,tr = self.update(xold,yold,self.posx,self.posy),[self.posx,self.posy]
        return a,tr
        
    def down(self):
        xold,yold = self.posx,self.posy
        if self.a[self.posx+1,self.posy] ==1:
            print('udar v stenku!')
            a,tr = 'udar',[self.posx,self.posy]
        else:
            self.posx = self.posx+1
            a,tr = self.update(xold,yold,self.posx,self.posy),[self.posx,self.posy]
        return a,tr     
        
    def left(self):
        xold,yold = self.posx,self.posy
        if self.a[self.posx,self.posy-1] ==1:
            print('udar v stenku!')
            a,tr = 'udar',[self.posx,self.posy]
        else:
            self.posy = self.posy-1
            a,tr = self.update(xold,yold,self.posx,self.posy),[self.posx,self.posy]
        return a,tr    
        
    def right(self):
        xold,yold = self.posx,self.posy
        if self.a[self.posx,self.posy+1] ==1:
            print('udar v stenku!')
            a,tr = 'udar',[self.posx,self.posy]
        else:
            self.posy = self.posy+1
            a,tr = self.update(xold,yold,self.posx,self.posy),[self.posx,self.posy]
        return a,tr
    
    def act(self,key,transition):
        randm = random.random()
    #    if randm>0.9:
    #        key = self.get_rnd_next_state(transition)
    #    print(key,' key')    
        if key ==0:
            r,tr = self.up()
        if key ==1:
            r,tr = self.down()
        if key ==2:
            r,tr = self.left()
        if key ==3:
            r,tr = self.right()
        return r,tr
    
    def get_poss_next_states(self,transition,nnext=8):
  # given a state s and a feasibility matrix F
  # get list of possible next states
        if nnext==8:
            tr = transition[self.posx,self.posy]
            actions = [index for index,value in enumerate(tr) if value==1]
        if nnext!=8:
            if nnext==0:
                tr = transition[self.posx-1,self.posy]
                actions = [index for index,value in enumerate(tr) if value==1]
            if nnext==1:
                tr = transition[self.posx+1,self.posy]
                actions = [index for index,value in enumerate(tr) if value==1]
            if nnext==2:
                tr = transition[self.posx,self.posy-1]
                actions = [index for index,value in enumerate(tr) if value==1]
            if nnext==3:
                tr = transition[self.posx,self.posy+1]
                actions = [index for index,value in enumerate(tr) if value==1]
                
        return actions 
    def get_rnd_next_state(self,transition):
  # given a state s, pick a feasible next state
        poss_next_states = self.get_poss_next_states(transition)
        next_state = poss_next_states[np.random.randint(0,len(poss_next_states))]
        return next_state


class agent(Game):
    def __init__(self,h,w):
        Game.__init__(self,h,w,par1=0,par2=0,par3=0,th=0,tw=0)
        self.positivebag = []
        self.negativebag = []
        self.bags = []
        self.trajss = []
        
            
            
    def init2(self):
        self.DD = np.ones((self.h,self.w))
        self.ro = np.zeros((self.h,self.w))
        self.Qt = np.zeros((self.h,self.w,5))
        self.Qt1 = np.zeros((self.h,self.w,4))
        self.setI = []
        self.NeSetIfu = []
        self.B = np.ones((self.h,self.w))*7
        self.transition1 = np.zeros((self.h,self.w,4))
        self.xx,self.yy =0,0
        
        self.staticfilter = [[self.th,self.tw],[self.th-1,self.tw],[self.th+1,self.tw],[self.th,self.tw-1],[self.th,self.tw+1],[self.th+1,self.tw+1],[self.th-1,self.tw-1],[self.th+1,self.tw-1],[self.th-1,self.tw+1]]
        for i in range(self.h):
            self.staticfilter.append([i,0])
            self.staticfilter.append([i,1])
            self.staticfilter.append([i,self.w-1])
          #  self.staticfilter.append([i,self.w-2])
        for i in range(self.w):
            self.staticfilter.append([0,i])
            self.staticfilter.append([self.h-1,i])
            self.staticfilter.append([1,i])
         #   self.staticfilter.append([self.h-2,i])
            
    def DDf(self):
        self.DD = np.ones((self.h,self.w))
        for i in range(self.h):
            for j in range(self.w):
                for n in range(len(self.positivebag)):
                    if [i,j] in self.positivebag[n]:
                        sump = 1
                        for p in range(len(self.positivebag[n])):
                            sumbc=0
                            for k in range(2):
                                sumbc += pow((self.positivebag[n][p][k]-self.th),2)    
                            sump *= 1-math.exp(-sumbc)  
                        self.DD[i,j] *= (1- sump)

        for i in range(self.h):
            for j in range(self.w):
                for n in range(len(self.negativebag)):
                    if [i,j] in self.negativebag[n]:
                        sump = 1
                        for p in range(len(self.negativebag[n])):
                            sumbc=0
                            for k in range(2):
                                sumbc += pow((self.negativebag[n][p][k]-self.tw),2)    
                            sump *= 1-math.exp(-sumbc)  
                        self.DD[i,j] *= (sump)
        return self.DD        