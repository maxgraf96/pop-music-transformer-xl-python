# =============================================
# This file contains the data loading routines
# =============================================
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import utilities as utils
from constants import INPUT_LENGTH

# Get input length
x_len = INPUT_LENGTH
# Define folder structure
train_folder = './data/train'
save_folder = './data/remi'
save_path = 'data/remi/dump.pkl'
midi_paths = 'data/train/*.midi'


def generate_all_remi():
    """
    Loops over all MIDI files in the train folder, generates REMI representations and saves them to
    data/remi/dump.pkl
    :return: None
    """
    is_existing_segments = os.path.isfile(save_path)
    if is_existing_segments:
        print('Found existing segments. If you want to generate all MIDI files again, delete the dump.pkl file. Skipping...')
        return
    # Generate all
    init_data('REMI-my-checkpoint')
    print("Generated REMI representations for all MIDI files in data/train folder.")


def init_data(checkpoint_path, num_segments_limit=None):
    """
    # Main data loading routine:
    # 1) Check if REMI segments were already generated (so we don't have to do it every time). If so, load.
    # 2) Parse num_files_limit MIDI files and generate segments for them, store them in default folder (see top of file)
    :param checkpoint_path: Path to the checkpoint directory used for vocabulary
    :param num_segments_limit: Max number of segments to load
    :return: The remi segments of the parsed MIDI files
    """
    # If segments were already generated load them
    is_existing_segments = os.path.isfile(save_path)
    if is_existing_segments:
        print('Found existing segments, loading...')
        with open(save_path, 'rb') as input_file:
            segments = pickle.load(input_file)
        print('Existing segments loaded.')
        print()
        if num_segments_limit is None:
            return segments
        else:
            return segments[:num_segments_limit, :, :, :]
    # Generate new
    else:
        print('No segments dump found, generating...')
        segments = prepare_and_save_data(checkpoint_path, num_segments_limit)
        return segments


def prepare_and_save_data(checkpoint_path, num_files_limit):
    """
    Create and save segments to file system
    :param checkpoint_path: Path to the folder of the vocabulary dict (voc is stored in the same folder as checkpoint)
    :param num_files_limit: How many files to convert to REMI representation
    :return: The generated segments
    """
    # Skip if folder already has files
    if os.path.isfile(save_path):
        print("Folder ", save_folder, ' already contains files... stopping...')
        return

    segments = prepare_data(checkpoint_path, num_files_limit)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    print('Saving ', num_files_limit, ' data segments to ', save_path)
    with open(save_path, "wb") as output_file:
        pickle.dump(segments, output_file)
    print('Saved ', num_files_limit, ' data segments to ', save_path)
    return segments


def prepare_data(checkpoint_path, num_files_limit=None):
    """
    Create segments of REMI data
    :param checkpoint_path: Path to the folder of the vocabulary dict (voc is stored in the same folder as checkpoint)
    :param num_files_limit: Number of files to convert to REMI format
    :return: The generated segments
    """
    if num_files_limit is None:
        num_files_limit = sys.maxsize

    # Load event2word (vocabulary)
    prefix = '' if 'transformer' in os.getcwd() else 'transformer/'
    dictionary_path = prefix + '{}/dictionary.pkl'.format(checkpoint_path)
    event2word, word2event = pickle.load(open(dictionary_path, 'rb'))

    print()
    print("=== Preparing data ===")
    print()
    print()
    print("=== Extracting events ===")
    print()
    all_events = []
    counter = 1
    # Get total number of files
    p = Path('data/train')
    total_files = len(list(p.glob('**/*')))
    p = Path('data/train')
    for path in p.glob('**/*.midi'):
        print("Extracting events from MIDI file ", counter, " out of ", total_files)
        events = extract_events(checkpoint_path, 'data/train/' + path.name)
        all_events.append(events)
        counter += 1
        if counter == num_files_limit:
            break

    # Get all words
    all_words = []
    for events in all_events:
        words = []
        for event in events:
            e = '{}_{}'.format(event.name, event.value)
            if e in event2word:
                words.append(event2word[e])
            else:
                # OOV
                if event.name == 'Note Velocity':
                    # replace with max velocity based on our training data
                    words.append(event2word['Note Velocity_21'])
                else:
                    # something is wrong
                    # you should handle it for your own purpose
                    print('something is wrong! {}'.format(e))
        all_words.append(words)
    # Convert words to segments of training data
    group_size = 5
    segments = []
    for words in all_words:
        pairs = []
        for i in range(0, len(words) - x_len - 1, x_len):
            x = words[i:i + x_len]
            y = words[i + 1:i + x_len + 1]
            pairs.append([x, y])
        pairs = np.array(pairs)
        # abandon the last
        for i in np.arange(0, len(pairs) - group_size, group_size * 2):
            data = pairs[i:i + group_size]
            if len(data) == group_size:
                segments.append(data)
    segments = np.array(segments)
    return segments


def extract_events(checkpoint_path, input_path):
    """
    Extract REMI events from MIDI data
    :param checkpoint_path: Path to checkpoint folder
    :param input_path: Path to MIDI file
    :return: The REMI events
    """
    # Get notes and tempo
    note_items, tempo_items = utils.read_items(input_path)
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end
    # Get chords
    chord_items = utils.extract_chords(note_items)
    items = chord_items + tempo_items + note_items
    groups = utils.group_items(items, max_time)
    # Convert to REMI representation
    events = utils.item2event(groups)
    return events