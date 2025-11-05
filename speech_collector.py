import os
from random import randint

import sounddevice as sd
import soundfile as sf
DIR = 'test/'
name_file = open("NAMES.txt")
names = [name.strip() for name in name_file.readlines()]
names_dict = dict.fromkeys(names, 0)
name_file.close()
for n in names:
    names_dict[n] = 0

TARGET = 5

# count existing recordings
for filename in os.listdir(DIR):
    if filename[:-7].title() in names and filename[-4:] == ".wav":
        name = filename[:-7].title()
        names_dict[name] += 1

# remove any completed names
for k, v in names_dict.items():
    if v >= TARGET:
        names.remove(k)
        print(f' ✅  {k}')
    else:
        print(f"{v:>3} {k}")

FS = 16000
SECONDS = 3

running = len(names) > 0
random = False
while running:
    name = 'AAAAAAAA'
    if random:
        while names_dict[name] >= TARGET and len(names) > 0:
            names.remove(name)  # remove any completed names
            print(f'{name} is completed')
            try:
                name = names[randint(0, len(names) - 1)]
            except IndexError:
                print('no names left')
            except:
                print('something went wrong')
        if len(names) == 0:
            print('All names complete')
            sd.wait()
            break
        cmd = input(f"{name} ({names_dict[name]}/{TARGET})").strip()
    else:
        while name not in names and running:
            name = input('please enter a name: ').strip().title()
            if name == 'Exit':
                cmd = 'exit'
                break
            cmd = ''
    if cmd == "":  # enter to record, 'p' to play recording after
        # record
        sd.stop()
        r = sd.rec(SECONDS * FS, samplerate=FS, channels=1)
        print(f"RECORDING...")
        sd.wait()

        # save recording "name123.wav"
        r_norm = 0.99 * r / max(abs(r))
        out_name = f"{DIR}{name.lower()}{names_dict[name] + 1:03}.wav" # stored like 'name001.wav'
        sf.write(out_name, r_norm, FS)
        sd.play(r_norm, FS)
        names_dict[name] += 1
    else:
        running = False
print("Stopping...")