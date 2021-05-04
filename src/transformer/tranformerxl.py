# =============================================
# This file contains the model code as well as training and generation routines
# =============================================

import os
import pickle
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import TransfoXLConfig, TransfoXLForSequenceClassification

import utilities as utils
from data_loading import generate_all_remi, extract_events, init_data
from dataset import CustomDataset
from label_smoothing_loss import LabelSmoothingLoss

# Seed for numpy random genrator
SEED = 987654321
# Input length of the transformer-xl model
input_length = 512
# Number of tokens for the model
n_token = 308

# Set hyperparameters
batch_size = 4
group_size = 5
# learning_rate = 0.00005 for 600 segments
# learning_rate = 0.00005
learning_rate = 0.00001

# Initialise the numpy and torch random number generators
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Get graphics device (cpu/gpu) and inform user
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print()
print("===")
print("DEVICE:", device)
print("===")
print()


def configure(pytorch_chkp=None):
    """
    Configure and load the model
    :param pytorch_chkp: Optional parameter for a pre-trained pytorch model checkpoint
    :return: The model
    """

    # Quick config for running different training configurations
    config1 = True
    if config1:
        n_head = 8
        n_layer = 12
        d_inner = 2048
        dropout = 0.1
    else:
        n_head = 6
        n_layer = 8
        d_inner = 1024
        dropout = 0.3

    # Initialise the Transformer XL configuration
    configuration = TransfoXLConfig(
        # Number of tokens
        n_token=n_token,
        # Number of self-attention layers for encoder and decoder
        n_layer=n_layer,
        # Number of attention heads for each attention layer in encoder
        n_head=n_head,
        # Length of the model's hidden states
        d_model=input_length,
        # Dimensionality of the model's heads
        d_head=input_length // n_head,
        # Inner dimension in feed-forward layer
        d_inner=d_inner,
        # Dropout probability
        dropout=dropout,
        # Dropout for attention probabilities
        dropatt=dropout,
        # Length of the retained previous heads
        mem_len=input_length,
        # Dimensionality/length of embeddings
        d_embed=input_length,
        # Length of target logits for classification
        tgt_len=input_length,
        # Length of the extended context
        ext_len=input_length,
        # Cutoffs for the adaptive softmax
        cutoffs=[],
        # Divident value for adapative input and softmax
        div_val=-1,
        # Use the same positional embeddings after clamp_len
        clamp_len=-1,
        # Whether to use the same attention length for all tokens
        same_length=False,
        # Number of samples in the sampled softmax
        sample_softmax=1,
        # Tie encoder weights to decoder weights
        tie_weight=True,
        tie_encoder_decoder=True,
        tie_word_embeddings=True,
        # Tie encoder biases to decoder biases
        untie_r=True,
        # Number of labels used for classification in the last layer
        num_labels=308,
        proj_share_all_but_first=False,
        # Make sure that this is greater than n_token!
        pad_token_id=309
    )

    # Initialise the model from the configuration
    model = TransfoXLForSequenceClassification(configuration)

    # Load a pre-trained checkpoint if it exists
    if pytorch_chkp is not None:
        model.load_state_dict(torch.load(pytorch_chkp, map_location=device))
        print("Loaded model checkpoint ", pytorch_chkp)

        # Apply model quantisation to massively speed up inference during testing/generating on CPUs
        if device.type != 'cuda':
            # Block the warning
            import warnings
            warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
            # Only quantise testing model, not during training.
            # Also, for some reason quantisation doesn't play well with all Nvidia GPUs
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear,
                 torch.nn.Softmax,
                 torch.nn.Embedding,
                 torch.nn.Dropout},
                dtype=torch.qint8
            )

    return model.to(device)  # Set model to graphics device (cpu/gpu)


def train(model, num_epochs=50, num_segments=600):
    """
    The training routine
    :param model: The transformer-xl model
    :param num_epochs: Number of training epochs
    :param num_segments: Number of REMI data segments to train with
    :return: None
    """
    # Log all numpy errors
    np.seterr('raise')

    # Run evaluation with validation set every n batches
    validation_every_n_batches = 10
    # Save model checkpoint every n epochs
    save_model_every_n_epochs = 10

    # Generate all REMI representations if they do not exist
    # This will generate remi segments and save them to a folder if they don't exist,
    # Otherwise just load pickled segments generated before
    generate_all_remi()

    # Prepare training data
    data = init_data(checkpoint_path='REMI-my-checkpoint', num_segments_limit=num_segments)
    dataset = CustomDataset(data)
    print("Dataset initialised.")

    # Train-validation split (0.1 means 90% training, 10% validation data)
    validation_split = 0.1
    # Whether to shuffle the data before training
    shuffle_dataset = True
    # Seed for randomly shuffling dataset
    random_seed = 42

    # Create data indices for training and validation splits
    dataset_size = len(dataset)
    print("Dataset length: ", dataset_size, " segments.")
    indices = list(range(dataset_size))
    # Get split index
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    from torch.multiprocessing import set_start_method
    # Try to use multiprocessing to speed up data loading
    try:
        set_start_method('spawn')
        num_workers = 1
    except RuntimeError:
        num_workers = 0
        pass

    # Create train and validation data loaders
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler,
                                               drop_last=True,
                                               num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler,
                                                    drop_last=True,
                                                    num_workers=0)

    # Helper field for time logging
    st = time.time()

    # Set model to training mode
    model.train()
    # Initialise ADAM optimiser and CosineAnnealing scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=400000,
                                                     eta_min=learning_rate * 0.004)

    # Save train and validation losses for evaluation
    train_losses = {}
    val_losses = {}

    # Initialise counter for number of completed batches
    num_completed_batches = 0

    # Initialise transformer memories
    mems = None

    # Set loss function
    criterion = LabelSmoothingLoss()

    # Run training loop for num_epochs iterations
    for epoch in range(num_epochs):
        # Initialise lists to keep losses for each epoch
        epoch_train_losses = []
        epoch_val_losses = []
        # Iterate over each batch
        for batch_idx, data in enumerate(train_loader):
            # Get the data segments
            segments = data['segments']
            # Further split the segments up into units that can be passed to the model
            for j in range(group_size):
                # Get data from segments
                segment_x = segments[:, j, 0, :]
                segment_y = segments[:, j, 1, :]

                # Uneven batches should never happen in the current implementation
                if segment_x.shape[0] is not batch_size or segment_y.shape[0] is not batch_size:
                    print()
                    print("xxx")
                    print("Uneven batch, skipping")
                    print("xxx")
                    print()

                # define input and target
                data = segment_x
                target = segment_y[:, -1]

                # Pass through model and get return values
                ret = model(input_ids=data, mems=mems)

                # Get logits and new memories from returned values
                logits = ret.logits
                new_mems = ret.mems

                # Calculate loss
                loss = criterion(logits, target)

                # Update memories
                mems = new_mems

                # Backpropagate
                loss.backward()
                # Step optimiser and scheduler
                optimizer.step()
                scheduler.step()
                # Reset gradients
                model.zero_grad()
                optimizer.zero_grad()

                # Take mean loss and print
                loss = loss.float().mean().type_as(loss)
                loss_mean = loss.float().mean()
                if device.type == 'cuda':
                    epoch_train_losses.append(loss_mean.cpu().detach().numpy())
                else:
                    epoch_train_losses.append(loss_mean.detach().numpy())
                print('>>> Epoch: {}, Loss: {:.5f}, Time: {:.2f}'.format(epoch, loss_mean, time.time() - st))

            # Increment number of completed batches
            num_completed_batches = num_completed_batches + 1
            # If number of completed batches reaches evaluation limit, evaluate model with validation set
            # This routine is similar to the one above, just with the validation data instead of training data
            if num_completed_batches % validation_every_n_batches == 0:
                print("===")
                print("Running on validation set...")
                print("===")
                # Switch model to evaluation mode
                model.eval()
                with torch.no_grad():
                    # Reset memories
                    mems = None
                    # Reset completed batches
                    num_completed_batches = 0
                    total_val_loss = 0.0
                    n_losses = 0
                    for batch_idx, data in enumerate(validation_loader):
                        segments = data['segments']
                        for j in range(group_size):
                            segment_x = segments[:, j, 0, :]
                            segment_y = segments[:, j, 1, :]

                            data = segment_x
                            target = segment_y
                            target = target[:, -1]
                            ret = model(input_ids=data, mems=mems)
                            loss = criterion(ret.logits, target)
                            loss = loss.float()
                            if device.type == 'cpu':
                                loss_float = loss.detach().numpy()
                            else:
                                loss_float = loss.cpu().detach().numpy()
                            total_val_loss += loss_float
                            n_losses = n_losses + 1

                    total_val_loss = total_val_loss / n_losses
                    epoch_val_losses.append(total_val_loss)
                    print("Mean loss on validation set: ", total_val_loss)
                    # Early stop if mean validation loss is below 0.1
                    if total_val_loss < 0.1:
                        print("Early stopping with loss ", total_val_loss)
                        save_losses(train_losses, val_losses)
                        return
                    # Set model back to training
                    model.train()

        # Save mean train and validation losses for epoch
        mean_train_loss = np.mean(epoch_train_losses)
        train_losses[epoch] = mean_train_loss
        # Print losses
        print()
        print("===")
        print("Train losses: ", train_losses)

        # Only print validation losses if they exist yet
        if len(epoch_val_losses) > 0:
            mean_val_loss = np.mean(epoch_val_losses)
            val_losses[epoch] = mean_val_loss
            print("Validation losses: ", val_losses)
        print("===")
        print()

        # Save model checkpoint to file if it is time
        if epoch > 0 and epoch % save_model_every_n_epochs == 0:
            save_model(model)

    # Save losses to pkl file
    save_losses(train_losses, val_losses)


def save_losses(train_losses, val_losses):
    """
    Saves losses to .pkl file
    :param train_losses: Dict of training losses
    :param val_losses: Dict of validation losses
    :return:
    """
    with open('evaluation/train_losses.pkl', "wb") as output_file:
        pickle.dump(train_losses, output_file)
    with open('evaluation/val_losses.pkl', "wb") as output_file:
        pickle.dump(val_losses, output_file)


def temperature_sampling(logits, temperature, topk):
    """
    Helper function that can be used to apply the concept of
    temperature to the output of the model during generation.
    See https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc
    for a detailed explanation.
    Function originally from https://github.com/YatingMusic/remi
    :param logits: Model output logits
    :param temperature: Temperature is used to increase the probability of probable tokens
    :param topk: Only sample from the top k candidates
    :return:
    """
    logits = logits.detach().numpy()
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    if topk == 1:
        prediction = np.argmax(probs)
    else:
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:topk]
        candi_probs = [probs[i] for i in candi_index]
        # normalize probs
        candi_probs /= sum(candi_probs)
        # choose by predicted probs
        prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return prediction


def generate(model, n_target_bar, temperature, topk, output_path, prompt=None, counter=0):
    """
    Generate and save a piece of MIDI
    :param model: The transformer-xl pytorch model
    :param n_target_bar: Number of bars to generate
    :param temperature: See function above
    :param topk: See function above
    :param output_path: The output path of the generated MIDI file
    :param prompt: Optional: Prompt the system with a MIDI input file
    :param counter: Number of MIDI file that is currently generated with respect to number of players in frontend (1, 2 or 3)
    :return: None
    """
    # Set model to evaluation mode
    model.eval()
    # Set batch size to 1 for generation
    batch_size = 1
    # Load event2word and word2event dicts (transformer vocabulary)
    prefix = '' if 'transformer' in os.getcwd() else 'transformer/'
    dictionary_path = prefix + '{}/dictionary.pkl'.format('REMI-my-checkpoint')
    event2word, word2event = pickle.load(open(dictionary_path, 'rb'))
    if prompt:
        # Load prompt and start new bar
        events = extract_events('REMI-my-checkpoint', prompt)
        words = [[event2word['{}_{}'.format(e.name, e.value)] for e in events]]
        words[0].append(event2word['Bar_None'])
    else:
        # Generating from scratch! Initialise sequence
        words = []
        for _ in range(batch_size):
            # Init new bar
            ws = [event2word['Bar_None']]
            # Set tempo
            tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
            tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
            # Load some random notes / chords
            chords = [v for k, v in event2word.items() if 'Chord' in k]
            ws.append(event2word['Position_1/16'])
            ws.append(np.random.choice(chords))
            ws.append(event2word['Position_1/16'])
            ws.append(np.random.choice(tempo_classes))
            ws.append(np.random.choice(tempo_values))
            words.append(ws)

    # Get length of token sequence
    original_length = len(words[0])
    # Is it the first run in generation loop below?
    initial_flag = 1
    # Number of bar we're currently generating, starts with 0
    current_generated_bar = 0
    # Reset memories
    mems = None
    # Index of generation iteration we're currently in
    generation_idx = 0

    # Generate bars until "n_target_bar" bars have been generated
    while current_generated_bar < n_target_bar:
        # If initial load all tokens into model input
        if initial_flag:
            model_input = np.zeros((batch_size, original_length))
            for b in range(batch_size):
                for z, t in enumerate(words[b]):
                    model_input[b][z] = t
            initial_flag = 0
        # If > 0th iteration of the generation loop only use last token as model input
        else:
            model_input = np.zeros((batch_size, 1))
            for b in range(batch_size):
                model_input[b][0] = words[b][-1]

        # Create data and target (labels) from model input
        data = torch.tensor(model_input, dtype=np.long).to(device)
        target = torch.tensor(model_input, dtype=np.long).to(device)[:, -1]

        # Pass data to model
        ret = model(input_ids=data, mems=mems, labels=target)
        # Get logits and new memories
        logits = ret.logits.cpu()
        new_mems = ret.mems

        # Set new memories
        mems = new_mems

        # Pick last word logit and convert to token word
        logit = logits[-1]
        word = temperature_sampling(
            logits=logit,
            temperature=temperature,
            topk=topk)
        words[0].append(word)
        # If bar event (only works for batch_size=1, so don't generate with batch_size > 1 currently)
        if word == event2word['Bar_None']:
            current_generated_bar += 1
        # Remedy to model not learning bar separations:
        # Attach it manually after a certain length
        if len(words[0]) % 100 == 0 and generation_idx != 0:
            words[0].append(event2word['Bar_None'])
            current_generated_bar += 1
            print("Stopping at " + str(len(words[0])) + " tokens, no end of bar token found...")
        generation_idx += 1

    # Write MIDI
    print("Writing MIDI...")
    if prompt:
        utils.write_midi(
            words=words[0][original_length:],
            word2event=word2event,
            output_path=output_path,
            prompt_path=prompt,
            counter=counter)
    else:
        utils.write_midi(
            words=words[0],
            word2event=word2event,
            output_path=output_path,
            prompt_path=None,
            counter=counter)


def save_model(model):
    """
    Save the model to './REMI-my-checkpoint/model.pt'
    :param model: The model to save
    :return: None
    """
    path = 'REMI-my-checkpoint/model.pt'
    torch.save(model.state_dict(), path)
    print('Model saved to ', path)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Test run configuration, currently not in use
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# if __name__ == "__main__":
    # Configure model
    # model = configure('REMI-my-checkpoint/model.pt')
    model = configure()

    # Train
    # train(model, num_epochs=5, num_segments=10)

    # Save checkpoint
    # save_model(model)

    # generate(model,
    #          n_target_bar=1,
    #          temperature=1.0,
    #          topk=1,
    #          output_path='./result/from_scratch.mid',
    #          prompt=None)
