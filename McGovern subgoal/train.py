import numpy as np
import random
import math
from mcgovern import Game
from mcgovern import agent
#from setup_flags import set_up

#FLAGS = set_up()

from tkinter import *
from tkinter import ttk
import time

def draw():
    global canvas
    for i in range(ag.a.shape[0]):
        for j in range(ag.a.shape[1]):

            if ag.a[i][j] != 1:
                color = "white"
            else:
                color = "black"
            id_num = i * len(ag.a[0]) + j + 1
         #   print(id_num)
            canvas.itemconfig(id_num,fill=color)
    state = ag.posx*ag.w+ag.posy    
    canvas.itemconfig(state+1,fill="blue")
    goal = ag.th*ag.w+ag.tw 
    canvas.itemconfig(goal+1,fill="yellow")  
    subgoal = ag.xx*ag.w+ag.yy 
    canvas.itemconfig(subgoal+1,fill="red")
    root.update()
  #  time.sleep(0.1)



ag = agent(10,10)
ag.stenka(5,4,6)
ag.tcel(8,8)
ag.init2()
ag.reset()


pixel_width = 480
block_length = pixel_width / ag.w
root = Tk()
root.title("Grid World")
canvas = Canvas(root,width = "500",height = "500")
canvas.grid(column=0, row=0, sticky=(N, W, E, S))
for i in range(ag.h):
    for j in range(ag.w):
        x_1 = 10 + block_length * j
        y_1 = 10 + block_length * i
        x_2 = x_1 + block_length
        y_2 = y_1 + block_length

        if ag.a[i][j] != 1:
            color = "white"
        else:
            color = "black"

        canvas.create_rectangle(x_1,y_1,x_2,y_2,fill=color)

ag = agent(10,10)
ag.stenka(5,4,6)
ag.tcel(8,8)
ag.init2()
for epochs in range(3000):
    ag.reset()
    bag = []
    trajs = []
    ag.transition1 = np.zeros((ag.h,ag.w,4))
    ag.transition[:,:,4] = 0
    
    for i in range(ag.h):
        for j in range(ag.w):
            if ag.B[i,j]==0:
                ag.transition[i,j,4]=1
                if ag.transition[i,j,0]==1 and [i-1,j] in ag.setI:
                    ag.transition1[i,j,0]=1

                if ag.transition[i,j,1]==1 and [i+1,j] in ag.setI:
                    ag.transition1[i,j,1]=1

                if ag.transition[i,j,2]==1 and [i,j-1] in ag.setI:
                    ag.transition1[i,j,2]=1

                if ag.transition[i,j,3]==1 and [i,j+1] in ag.setI:
                    ag.transition1[i,j,3]=1
                    
    ag.transition[ag.xx,ag.yy,4]=0
    ag.transition1[ag.xx,ag.yy] = ag.transition[ag.xx,ag.yy,:4]                
    num_opt = 0    
    
    for i in range(50):
        actions = ag.get_poss_next_states(ag.transition)
        startOption = False
        vall = [value for index,value in enumerate(ag.Qt[ag.posx,ag.posy]) if index in actions]
        print(ag.posx,ag.posy, 'position',actions,vall)
        maxval = max(vall)
        maxindval = [index for index,value in enumerate(ag.Qt[ag.posx,ag.posy]) if (index in actions and value==maxval)]
        next_s = random.choice(maxindval)
        
        if next_s==0:
            next_x,next_y = ag.posx-1,ag.posy
        if next_s==1:
            next_x,next_y = ag.posx+1,ag.posy
        if next_s==2:
            next_x,next_y = ag.posx,ag.posy-1
        if next_s==3:
            next_x,next_y = ag.posx,ag.posy+1
        if next_s==4:
            Rr = 0
            iii = 0
            mmax_Q = -9999
            oldX = ag.posx
            oldY = ag.posy
            while [ag.posx,ag.posy] in ag.setI and [ag.posx,ag.posy]!=[ag.xx,ag.yy] and iii<10:
                num_opt+=1

                startOption = True
                actions = ag.get_poss_next_states(ag.transition1)

                vall = [value for index,value in enumerate(ag.Qt1[ag.posx,ag.posy]) if index in actions]
                maxval = max(vall)
                maxindval = [index for index,value in enumerate(ag.Qt1[ag.posx,ag.posy]) if (index in actions and value==maxval)]
                next_s = random.choice(maxindval)
                print(ag.posx,ag.posy, 'POSITION',next_s,actions,vall)
                if next_s==0:
                    next_x,next_y = ag.posx-1,ag.posy
                if next_s==1:
                    next_x,next_y = ag.posx+1,ag.posy
                if next_s==2:
                    next_x,next_y = ag.posx,ag.posy-1
                if next_s==3:
                    next_x,next_y = ag.posx,ag.posy+1
                poss_next_next_states = ag.get_poss_next_states(ag.transition1,nnext=next_s)
                max_Q = -9999
                
                for j in range(len(poss_next_next_states)):
                    nn_s = poss_next_next_states[j]
                    q = ag.Qt1[next_x,next_y,nn_s]
                    if q > max_Q:
                        max_Q = q   
                        
                R = ag.reward[ag.posx,ag.posy,next_s]
                if next_s==0 and iii>0:
                    if [(ag.posx-1),(ag.posy)]==[ag.xx,ag.yy]:
                        R+=1
                if next_s==1 and iii>0:
                    if [(ag.posx+1),ag.posy]==[ag.xx,ag.yy]:
                        R+=1
                if next_s==2 and iii>0:
                    if [ag.posx,(ag.posy-1)]==[ag.xx,ag.yy]:
                        R+=1
                if next_s==3 and iii>0:
                    if [ag.posx,(ag.posy+1)]==[ag.xx,ag.yy]:
                        R+=1
                if R>5:
                    print('ADD REWARD ',ag.posx,ag.posy,next_s)
                Rr+=R*pow(0.9,iii)
                if max_Q>mmax_Q:
                    mmax_Q = max_Q
                ag.Qt1[ag.posx,ag.posy][next_s] = ((1 - ag.lr) * ag.Qt1[ag.posx,ag.posy][next_s]) + (ag.lr * (R +(ag.gamma * max_Q)))
                randm = random.random()
                if randm>0.9:
                    next_s = ag.get_rnd_next_state(ag.transition1)
                r,tr = ag.act(next_s,ag.transition1)
                draw()
                trajs.append(next_s)
                bag.append(tr)
                iii+=1
            R = Rr
            max_Q = mmax_Q
            ag.Qt[oldX,oldY][4] = ((1 - ag.lr) * ag.Qt[oldX,oldY][4]) + (ag.lr * (R +(ag.gamma * max_Q)))
            ag.transition[:,:,4] = 0  

            
        if startOption==False:    
                poss_next_next_states = ag.get_poss_next_states(ag.transition,nnext=next_s)    

                max_Q = -9999
                for j in range(len(poss_next_next_states)):
                    nn_s = poss_next_next_states[j]
                    q = ag.Qt[next_x,next_y,nn_s]
                    if q > max_Q:
                        max_Q = q
                R = ag.reward[ag.posx,ag.posy,next_s]
                ag.Qt[ag.posx,ag.posy][next_s] = ((1 - ag.lr) * ag.Qt[ag.posx,ag.posy][next_s]) + (ag.lr * (R +(ag.gamma * max_Q)))


                randm = random.random()
                if randm>0.9:
                    next_s = random.randint(0,3)
                r,tr = ag.act(next_s,ag.transition)
                draw()
                trajs.append(next_s)
                bag.append(tr)
        if r=='win':
            ag.positivebag.append(bag)
            break        
    if r!='win':
        ag.negativebag.append(bag)
    ag.bags.append(bag)
    ag.trajss.append(trajs)
    ag.DD = ag.DDf()
    ag.DD = np.log(ag.DD)*(-1)  
    ind = np.where(ag.DD==ag.DD.max())
    x_y_coords =  zip(ind[0], ind[1])
 #   print('!!!!!!!!!!!!!!',ind,ag.DD.max())
    ag.setI = []
    ag.NeSetI = []
    ag.B = np.ones((ag.h,ag.w))*7
    options= 0
    maxro = 0
    dooption = False
    for x,y in x_y_coords:
        if ([x,y] not in ag.staticfilter) and [x,y] in [item for sublist in ag.bags for item in sublist]:
            ag.ro[x,y]+=1
            ag.B = np.ones((ag.h,ag.w))*7
            
            if (ag.ro[x,y]>=4 and ag.ro[x,y]>=maxro):
                maxro = ag.ro[x,y]
                ag.xx = x
                ag.yy = y
                dooption = True
                options +=1
                ag.setI = []

                for bbag,ttraj in zip(ag.bags[-20:],ag.trajss[-20:]):

                    if [ag.xx,ag.yy] in bbag[2:]:
                        NeSetIfu = []
                        for i2,i3 in enumerate(zip(bbag,ttraj)):
                            inst = i3[0]
                            insttr = i3[1]
                            NeSetIfu.append(inst)
                            if inst not in ag.setI:
                                ag.setI.append(inst)
                            if [ag.xx,ag.yy] == inst:
                                break    
                        ag.NeSetI.append(NeSetIfu)

    if dooption:    
        for ib in range(ag.h):
            for jb in range(ag.w):
                if [ib,jb] not in ag.setI:
                    ag.B[ib,jb] = 1
                else:
                    ag.B[ib,jb] = 0
        ag.B[ag.xx,ag.yy] = 0            
        print(ag.B)
        print(ag.xx,ag.yy,' XX YY')
        ag.Qt[:,:,4]*=(1-ag.B)
    ag.ro =ag.ro*0.8
    
  #  break     
root.destroy()