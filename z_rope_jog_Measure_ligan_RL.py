import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
from sensors.ligan import LiganSensor
import numpy as np
from collections import deque
import threading
import socket
from RLmylib.RL_lib import RealEnv, ActorCritic, PPOTrainer
import torch
import gym


Rope_S = 0
Touch_S = 0
cur_pos_abs = 0
Weight = 0

buffer_dyn_Srope = deque([0.0]*3, maxlen=3)
buffer_weight_Srope = deque([0.0]*30, maxlen=50)
buffer_dyn_Stouch = deque([0.0]*3, maxlen=3)
buffer_weight_Stouch = deque([0.0]*30, maxlen=50)
buffer_rising_CurPos = deque([0]*5, maxlen=5)
rising_slope = 0

def read_cur_pos(myXYZ, InitPos):
    global cur_pos_abs
    global buffer_rising_CurPos
    global rising_slope
    while True:
        cur_pos_abs = myXYZ.Get_Pos(3)+InitPos
        buffer_rising_CurPos.append(cur_pos_abs)
        x = np.arange(len(buffer_rising_CurPos))*1000
        rising_slope, _ = np.polyfit(x, -np.array(buffer_rising_CurPos), 1)
        time.sleep(0.001)

def read_rope_sensor():
    global Rope_S
    global buffer_dyn_Srope
    global buffer_weight_Srope
    Srope = DaYangSensor('/dev/ttyUSB0',0)
    try:
        while True:
            Rope_S = Srope.read_angles()
            buffer_dyn_Srope.append(Rope_S)
            buffer_weight_Srope.append(Rope_S)
            time.sleep(0.001)
    except KeyboardInterrupt:
        Srope.close()
        print("Ctrl-C is pressed!")

def touch_sensor_server(host='0.0.0.0', port=65432):
    global Touch_S
    global buffer_dyn_Stouch
    global buffer_weight_Stouch
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server started, waiting for connection on {port}...")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                msg = data.decode()
                Touch_S = int(msg.split()[-1])
                buffer_dyn_Stouch.append(Touch_S)
                buffer_weight_Stouch.append(Touch_S)

def VelCont(Vgoal_N,Vgoal,myXYZ):
    diff=(Vgoal_N-Vgoal)*0.01
    Max_diff=15
    diff = np.clip(diff, -Max_diff, Max_diff)
    Max_Vel=4000
    Vgoal = np.clip(Vgoal+diff, -Max_Vel, Max_Vel)
    myXYZ.AxisMode_Jog(3,30,Vgoal)
    return Vgoal

def main():
    try:
        myXYZ = xyz_utils()
        print('Start Enable')
        myXYZ.OpenEnableZero_ALL()
        InitPos=myXYZ.Safe_Jog()
        threading.Thread(target=read_rope_sensor, args=(), daemon=True).start()
        threading.Thread(target=touch_sensor_server, args=(), daemon=True).start()
        threading.Thread(target=read_cur_pos, args=(myXYZ,InitPos,), daemon=True).start()
        while Touch_S==0:
            time.sleep(0.1)
            print('Waiting Touch Data!!')
        
        Vgoal=0
        Vgoal_N=0
        Touch_valve=700
        Pnum=0
        model_path = "/home/mo/RoboGloveWS/RoboGloveProj/RLmylib/ppo_rope_lift_final.pth"
        while True:
            time.sleep(0.001)
            if np.mean(buffer_dyn_Stouch)>Touch_valve and Weight==0:
                print('Enter Measurement!!!')
                mode=1 # Measure Mode
                cur_time = time.time()
                while Rope_S<500 and (time.time()-cur_time)<1:
                    # Vgoal=2000+0.005*Touch_S
                    Vgoal=2000
                    myXYZ.AxisMode_Jog(3,30,Vgoal)
                    print('Waiting Lifting, Rope_S:',Rope_S, 'Elapsed time:',time.time()-cur_time)
                    time.sleep(0.01)
                time.sleep(0.5)
                if Rope_S>500:
                    # Weight = 3*np.mean(buffer_weight_Stouch) # put touch senser on top suface, can be a baseline
                    Weight = np.mean(buffer_weight_Srope) + 0.9*(np.mean(buffer_weight_Stouch)-650)
                    print('Measured Success! Weight:', Weight)
                else:
                    Vgoal=0
                    myXYZ.AxisMode_Jog(3,30,Vgoal)
                    time.sleep(0.5)
                    print('Measured Failure: Over Time!')
            elif np.mean(buffer_dyn_Stouch)<=Touch_valve:
                mode=3 # Lossen Mode
                Weight=0
                err=40-np.mean(buffer_dyn_Srope)
                Vgoal_N=40*err
                Vgoal = VelCont(Vgoal_N,Vgoal,myXYZ)
            else:
                mode=2 # Load Mode
                env = RealEnv(buffer_weight_Srope,buffer_weight_Stouch,buffer_rising_CurPos)
                policy = ActorCritic(env.state_dim)
                policy.load_state_dict(torch.load(model_path, weights_only=True))
                trainer = PPOTrainer(policy)
                try:
                    count = 0
                    state = env.state
                    states, actions, rewards, log_probs = [], [], [], []
                    current_policy = trainer.latest_policy
                    while np.mean(buffer_dyn_Stouch)>Touch_valve:
                        if rising_slope > 2: # Update weight only when rising
                            state_tensor = torch.FloatTensor(state).unsqueeze(0)
                            with torch.no_grad():
                                dist, value = current_policy(state_tensor)
                                action = dist.sample()
                                log_prob = dist.log_prob(action)

                            Weight = abs(action)*3000
                            Weight = np.clip(Weight, 0, 3000)
                            err=Weight-np.mean(buffer_dyn_Srope)
                            if abs(err)<150:
                                err=0
                            Vgoal_N=15*err
                            Vgoal = VelCont(Vgoal_N,Vgoal,myXYZ)
                            time.sleep(0.05) # for acting

                            next_state, reward, done = env.step(buffer_weight_Srope,buffer_weight_Stouch,buffer_rising_CurPos)
                            states.append(state)
                            actions.append(action)
                            rewards.append(reward)
                            log_probs.append(log_prob)
                            state = next_state
                            count += 1
                            if count % 10 == 0:
                                print('Upload data', count/10)
                                trainer.data_queue.put((states, actions, rewards, log_probs), block=False) # 非阻塞方式添加数据，如果队列满则跳过
                                states, actions, rewards, log_probs = [], [], [], []
                                current_policy = trainer.latest_policy
                        else:
                            err=Weight-np.mean(buffer_dyn_Srope)
                            if abs(err)<150:
                                err=0
                            Vgoal_N=15*err
                            Vgoal = VelCont(Vgoal_N,Vgoal,myXYZ)
                        
                        print('Mode:',mode,
                            'ave_dyn_Srope:',int(np.mean(buffer_dyn_Srope)),
                            'ave_dyn_Stouch:',int(np.mean(buffer_dyn_Stouch)),
                            'Weight:',int(Weight),
                            'slope:', int(rising_slope),
                            )

                except KeyboardInterrupt:
                    print("Training interrupted")
                finally:
                    trainer.stop_event.set()
                    trainer.train_thread.join()
                    torch.save(trainer.latest_policy.state_dict(), model_path)

            Pnum += 1
            if Pnum % 29 == 0:
                print('Mode:',mode,
                      'ave_dyn_Srope:',int(np.mean(buffer_dyn_Srope)),
                      'ave_dyn_Stouch:',int(np.mean(buffer_dyn_Stouch)),
                    #   'Vgoal_N:',int(Vgoal_N),
                    #   'Vgoal:',int(Vgoal),
                    #   'diff:',int(diff),
                      'Weight:',int(Weight),
                      'slope:', int(rising_slope),
                      )
            # myXYZ.AxisMode_Jog(3,30,-2000)

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        sys.exit(0)

if __name__ == "__main__":
    main()


