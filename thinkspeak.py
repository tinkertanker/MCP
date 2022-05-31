#!/usr/bin/env python3

import socket
import os
import time
import urllib.request

# Upload number of people to ThinkSpeak

if __name__ == '__main__':

    from tkinter import Tk
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--peoplenum-filename', type=str,
                        help="Alternative path from which to read the number of people",
                        default='/dev/shm/millenia.txt')
    parser.add_argument('-n', '--hostname', type=str,
                        help="Override Hostname",
                        default='')

    args = parser.parse_args()

    succeeded = False
    peoplenum = 0
    starttime = time.time()

    while time.time() - starttime < 10: # keep trying for 10 seconds
        if os.path.exists(args.peoplenum_filename):
            try:
                with open(args.peoplenum_filename) as f:
                    peoplenum = int(f.readline())
                    print(f'read peoplenum of {peoplenum} from file')
                    succeeded = True
                    break
            except:
                pass

    if succeeded:
        hostname = args.hostname or socket.gethostname()
        with urllib.request.urlopen(f'https://api.thingspeak.com/update.json?api_key=5NO8KS8QNN4BR9WY&field{hostname[-1:]}={peoplenum}') as f:
            print(f.read().decode('utf-8'))
    else:
        print("Timed out.")

    
    