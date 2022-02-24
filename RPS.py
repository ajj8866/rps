from itertools import count
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from keras.models import load_model
import time


##########################################################################################################
##################### ROCK PAPER SCISSORS INITIAL FUNCTION ###############################################
##########################################################################################################

def rockPaperSciss(tlimit = 10):
    model = load_model(Path(os.getcwd(), 'converted_keras', 'keras_model.h5'))
    cap = cv2.VideoCapture(0)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    cpu_counter = ['Rock', 'Paper', 'Scissor']
    start_time = time.time()
    win_loss = []
    while True:
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, (224, 224), interpolation= cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127) - 1
        data[0] = normalized_image
        prediction = model.predict(data)
        print(prediction[0])
        cv2.imshow('frame', frame)
        prob_lim = 0.5
        choice_rps = np.random.choice(cpu_counter, size=1, replace=True)
        pred_list = [prediction[0][0], prediction[0][1], prediction[0][2], prediction[0][3]]
        ''' 
        For code below:
        1) Outer if condition in major loop checks which index yields to highest probability 
        value and whether such a probabilty is greater than the value specified by prob_lim
        2) Inner if condition compares randomly chosen computer selection from choices rock,
        paper scissors and compares against the selection made by player as inferred by machine.
        3) Should player win 1 is appended list instantiated on each iteration of while loop, should
        computer win -1 is appended and should there be a draw or player inferred to have done nothing
        0 is appended
        4) The values within the instantiated list are summed and should summed values be greater than
        0 player deduced to have won most round, should summed values be less than 0 computer inferred
        to have won most round and should sum be equal to zero both player and machine have won equal 
        number of rounds
        '''
        if (pred_list.index(max(pred_list)) == 0) and (prediction[0][0] > prob_lim):
            if choice_rps == 'Rock':
                print(f'Computer picked {choice_rps[0]} and you picked Rock ----> Draw')
                win_loss.append(0)
            elif choice_rps == 'Paper':
                print(f'Computer picked {choice_rps[0]} and you picked Rock ----> You Lose :(')
                win_loss.append(-1)
            elif choice_rps == 'Scissor':
                print(f'Computer picked {choice_rps[0]} and you picked Rock ----> You Win!')
                win_loss.append(1)
            else:
                print('...do something')
                win_loss.append(0)
        elif (pred_list.index(max(pred_list)) == 1) and (prediction[0][1] > prob_lim):
            if choice_rps == 'Rock':
                print(f'Computer picked {choice_rps[0]} and you picked Paper ----> You win!')
                win_loss.append(1)
            elif choice_rps == 'Paper':
                print(f'Computer picked {choice_rps[0]} and you picked Paper ----> Draw')
                win_loss.append(0)
            elif choice_rps == 'Scissor':
                print(f'Computer picked {choice_rps[0]} and you picked Paper ----> You Lose :(')
                win_loss.append(-1)
            else:
                print('...do something')
                win_loss.apend(0)
        elif (pred_list.index(max(pred_list)) == 2) and (prediction[0][2] > prob_lim):
            if choice_rps == 'Rock':
                print(f'Computer picked {choice_rps[0]} and you picked Scissor ----> You Lose :(')
                win_loss.append(-1)
            elif choice_rps == 'Paper':
                print(f'Computer picked {choice_rps[0]} and you picked Scissor ----> You Win!')
                win_loss.append(1)
            elif choice_rps == 'Scissor':
                print(f'Computer picked {choice_rps[0]} and you picked Scissor ----> Draw')
                win_loss.append(0)
            else:
                print('...do something')
                win_loss.append(0)
        else:
            print('...do something')
            win_loss.append(0)

        if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time > tlimit):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    if sum(win_loss) > 0:
        print('*'*20)
        print(f'You won {sum(win_loss)} more times than the computer beat you so you WIN!')
    elif sum(win_loss) < 0:
        print('*'*20)
        print(f'You lost {abs(sum(win_loss))} more times than the computer lost to you so you LOSE!')
    else:
        print('DRAW')
    return sum(win_loss)

##########################################################################################################
##################### FIRST TO THREE/ BEST OF THREE RPS FUNCTION #########################################
##########################################################################################################

def tournamentWinner(type = 'best_off',rounds = 3, round_len = 15):
    counter = 0
    # Code below for when best of option selected for type of tournament i.e. number of rounds fixed 
    if type == 'best_off':
        for i in range(3):
            winner = rockPaperSciss(round_len)
            if winner > 0:
                counter = counter + 1
            elif winner < 0:
                counter = counter - 1
        if counter > 0:
            print(counter)
            print('You win tournament')
        elif counter < 0:
            print(counter)
            print('You lose tournament')
        else:
            print(counter)
            print('Draw')
    # Code below for when 'first_to' option is selected for type of tournament i.e. tournament ends when either one
    # of computer or player wins number of games specified in rounds argument        
    elif type == 'first_to':
        you = 0
        cpu = 0
        while (you < rounds) and (cpu < rounds):
            winner = rockPaperSciss(round_len)
            if winner > 0:
                you = you + 1
            elif winner < 0:
                cpu = cpu + 1
        print('#'*30)
        print('Player total score: ', you)
        print('Computer total score: ', cpu)
        if you > cpu:
            print('You win!')
        else: 
            print('You lose :(')
        print('#'*30)
        
    

tournamentWinner(type='first_to',  rounds=2, round_len=10)