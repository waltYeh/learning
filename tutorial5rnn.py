from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import os
import time
emb_size = 8
hidden_size = 128  # size of hidden layer of neurons
seq_length = 64  # number of steps to unroll the RNN for, batch


def build_model(vocab_size, emb_size, hidden_size, batch_size):

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_size, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(hidden_size,return_sequences=True,stateful=True),#return_sequences ensures the dim of 3 layers the same
        tf.keras.layers.Dense(vocab_size)])
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    # Add a LSTM layer with 128 as size of hidden layer of neurons.
    # Add a Dense layer 
    return model
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
def generate_text(model, start_string_ix, ix_to_char, num_generate):
    # Evaluation step (generating text using the learned model)


    # Converting our start string to numbers (vectorizing)
    input_eval = [start_string_ix]
    print(input_eval)
    input_eval = tf.expand_dims(input_eval, 0)
    print(input_eval)
    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(ix_to_char[predicted_id])

    return (ix_to_char[start_string_ix] + ''.join(text_generated))

def main():
    # data I/O
    text = open('/home/yexin/Tensor/RNN/data/input.txt', 'r').read() # should be simple plain text file
    chars = list(set(text))
    data_size, vocab_size = len(text), len(chars)
    print('text has %d characters, %d unique.' % (data_size, vocab_size))
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    model = build_model(vocab_size, emb_size, hidden_size, batch_size=seq_length)

    model.summary()
    model.compile(optimizer='adagrad', loss=loss)
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    n, p = 0, 0
    n_updates = 0
    while True:

        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p+seq_length+1 >= len(text) or n == 0:
       #     hprev = np.zeros((hidden_size,1)) # reset RNN memory
       #     cprev = np.zeros((hidden_size,1))
            p = 0 # go from start of data
        inputs = [char_to_ix[ch] for ch in text[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in text[p+1:p+seq_length+1]]

        # sample from the model now and then
        

        history = model.fit(inputs, targets, epochs=1, batch_size=seq_length, callbacks=[checkpoint_callback])

        if n % 100 == 0:
            model_sample = build_model(vocab_size, emb_size, hidden_size, batch_size=1)
            model_sample.load_weights(tf.train.latest_checkpoint(checkpoint_dir)) 
            model_sample.build(tf.TensorShape([1, None]))
            sample_ix = generate_text(model_sample, inputs[0], ix_to_char,2000)
            txt = ''.join(ix for ix in sample_ix)
            print ('----\n %s \n----' % (txt, ))
            input("Press Enter to continue...")
        max_updates = 500000
        p += seq_length # move data pointer
        n += 1 # iteration counter
        n_updates += 1
        print(n)
        if n_updates >= max_updates:
            break
    # tf.train.latest_checkpoint(checkpoint_dir)

    # model2 = build_model(vocab_size, emb_size, hidden_size, batch_size=1)

    # model2.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    # model2.build(tf.TensorShape([1, None]))
    # print(generate_text(model2, start_string=u"ROMEO: "))


main()