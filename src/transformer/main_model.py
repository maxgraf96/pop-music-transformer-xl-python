# =============================================
# This file contains the major functions for interfacing with the model
# =============================================

import os
from shutil import copyfile
import mido

from tranformerxl import configure, generate
from constants import MODEL_PATH

# Check whether a pre-trained checkpoint exists
path_chkp = MODEL_PATH if os.path.exists(MODEL_PATH) else None


def init_transformer_model():
    """
    Initialise the transformer-xl model
    :return:
    """
    global model
    model = None
    if model is None:
        model = configure(path_chkp)


def generate_from_scratch():
    """
    Generate a bar of MIDI from scratch and save it to './result/generated_x.mid'
    :return: None
    """

    # Create paths for resulting midi files
    counter = 0
    path1 = 'result/generated_' + str(counter) + ".mid"
    path2 = 'result/generated_' + str(counter + 1) + ".mid"
    path3 = 'result/generated_' + str(counter + 2) + ".mid"
    paths = [path1, path2, path3]

    # generate three midi files from scratch and save them
    for i in range(3):
        generate(model,
                 n_target_bar=1,
                 temperature=1.2,
                 topk=5,
                 output_path=paths[i],
                 prompt=None,
                 counter=i)


def generate_from_conditioned(player_idx):
    """
    Generate a bar of MIDI from another MIDI file and save it to './result/generated_x.mid'
    :param player_idx: The player index from the frontend used to condition this generation
    :return: None
    """
    print("Generating from conditioned with player index ", str(player_idx))
    counter = 0

    path_condition_origin = 'result/generated_' + str(player_idx) + ".mid"
    path_condition_dest = 'result/baseline.mid'
    # Copy condition file
    copyfile(path_condition_origin, path_condition_dest)

    # Output path of the three new midi files
    path1 = 'result/generated_' + str(counter) + ".mid"
    path2 = 'result/generated_' + str(counter + 1) + ".mid"
    path3 = 'result/generated_' + str(counter + 2) + ".mid"
    paths = [path1, path2, path3]

    # generate three midi files from conditioned => overwrite generated for now
    for i in range(3):
        generate(model,
                 n_target_bar=1,
                 temperature=1.2,
                 topk=5,
                 output_path=paths[i],
                 prompt=path_condition_dest,
                 counter=i)


def generate_from_recorded():
    """
    Generate from a user-recorded MIDI file
    :return: None
    """

    counter = 0
    path_condition = 'result/my_recording.mid'

    # Note: When taking MIDI input created with js frontend we need to reverse the order of note_on and note_off
    # events in the midi file, since the js library apparently writes them in the "wrong" order
    # which means that the MIDI parser doesn't pick up any notes otherwise...
    try:
        # Load MIDI file created with JS
        mid = mido.MidiFile(path_condition)
        # Get its tracks
        track = mid.tracks[0]
        note_ons = []
        note_offs = []
        # Extract the note_on and note_off events
        for event in track:
            if event.type == 'note_on':
                note_ons.append(event)
            if event.type == 'note_off':
                note_offs.append(event)

        # Create a duplicate list and delete the note_on and note_off events
        del_list = [x for x in track if not x.type == 'note_on']
        del_list = [x for x in del_list if not x.type == 'note_off']

        # Re-append them in the correct order (first all note_ons then note_offs)
        del_list.extend(note_ons)
        del_list.extend(note_offs)
        # Finally, overwrite list in midi file and overwrite existing midi file
        mid.tracks[0] = del_list
        mid.save(path_condition)
    except Exception as e:
        print(e)
        print("Couldn't replace MIDI file while swapping note_on/note_off order. Can't generate from recorded...")
        return

    # Output path of the three new midi files
    path1 = 'result/generated_' + str(counter) + ".mid"
    path2 = 'result/generated_' + str(counter + 1) + ".mid"
    path3 = 'result/generated_' + str(counter + 2) + ".mid"
    paths = [path1, path2, path3]

    # generate three midi files from conditioned => overwrite generated for now
    for i in range(3):
        generate(model,
                 n_target_bar=1,
                 temperature=1.2,
                 topk=5,
                 output_path=paths[i],
                 prompt=path_condition,
                 counter=i)
