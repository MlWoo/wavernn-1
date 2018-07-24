"""
Run inference on a WAV file using a frozen WaveRNN model.
"""
import math
import os
import random
import time

import click
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

MEL_BANDS = 80
SAMPLE_RATE = 16000
SCALING = 0.185

# Frozen graph nodes.
OUTPUT_NODE = "Inference/Model/MuLawExpanding/mul_1:0"
INPUT_NODE = "IteratorGetNext:1"
TRAINING = "training:0"

# dump paramters
Weights_W = "Model/Wave-RNN/Weights_W/read:0"
Biases_W = "Model/Wave-RNN/Biases_W/read:0"
Weights_R = "Model/Wave-RNN/Weights_R/read:0"
Biases_R = "Model/Wave-RNN/Biases_R/read:0"

Weights_R_W = "Model/Relu-1/Affine/Weights/read:0"
Biases_R_W = "Model/Relu-1/Affine/Bias/read:0"

Weights_O_W = "Model/Output/Affine/Weights/read:0"
Biases_O_W = "Model/Output/Affine/Bias/read:0"

Weights_A_E = "Encoder/Affine/Weights/read:0"
Biases_A_E = "Encoder/Affine/Bias/read:0"

Moving_Var_1 =  "Encoder/Layer-1/batch_normalization/moving_variance/read:0"
Moving_Mean_1 =  "Encoder/Layer-1/batch_normalization/moving_mean/read:0"
Gamma_1 = "Encoder/Layer-1/batch_normalization/gamma/read:0"
Beta_1 = "Encoder/Layer-1/batch_normalization/beta/read:0"
Weights_1 = "Encoder/Layer-1/Conv1D/Weights/read:0"
Biases_1 = "Encoder/Layer-1/Conv1D/Bias/read:0"

Moving_Var_2 =  "Encoder/Layer-2/batch_normalization/moving_variance/read:0"
Moving_Mean_2 =  "Encoder/Layer-2/batch_normalization/moving_mean/read:0"
Gamma_2 = "Encoder/Layer-2/batch_normalization/gamma/read:0"
Beta_2 = "Encoder/Layer-2/batch_normalization/beta/read:0"
Weights_2 = "Encoder/Layer-2/Conv1D/Weights/read:0"
Biases_2 = "Encoder/Layer-2/Conv1D/Bias/read:0"

Moving_Var_3 =  "Encoder/Layer-3/batch_normalization/moving_variance/read:0"
Moving_Mean_3 =  "Encoder/Layer-3/batch_normalization/moving_mean/read:0"
Gamma_3 = "Encoder/Layer-3/batch_normalization/gamma/read:0"
Beta_3 = "Encoder/Layer-3/batch_normalization/beta/read:0"
Weights_3 = "Encoder/Layer-3/Conv1D/Weights/read:0"
Biases_3 = "Encoder/Layer-3/Conv1D/Bias/read:0"

# dump the results of key nodes
Residual = "Encoder/Residual/add:0"
Upsampleing =  "Encoder/UpsampleByRepetition/Reshape:0"

@click.command()
@click.argument("wav")
@click.option("--model", default="models/frozen.pb", help="Frozen graph")
@click.option("--output", default="outputs/audio.wav", help="Output WAV audio")
def inference(wav, model, output):
    """
    Converts an input WAV file to an 80-band mel spectrogram, then runs
    inference on the spectrogram using a frozen graph.

    Writes the output to a WAV file.
    """
    data, sr = librosa.core.load(wav, sr=SAMPLE_RATE, mono=True)
    print("Length of audio: {:.2f}s".format(float(len(data))/sr))

    spectrogram = compute_spectrogram(data, sr)
    #plot_spectrogram(spectrogram)

    audio = run_wavernn(model, spectrogram, output)
    run_wavernn(model, spectrogram, output)
    librosa.output.write_wav(output, audio, sr=SAMPLE_RATE)
    print("Wrote WAV file:", os.path.abspath(output))


def compute_spectrogram(audio, sr):
    """
    Converts audio to an 80-band mel spectrogram.

    Args:
        audio: Raw audio data.
        sr:    Audio sample rate in Hz.

    Returns:
        80-band mel spectrogram, a numpy array of shape [frames, 80].
    """
    spectrogram = librosa.core.stft(audio, n_fft=2048, hop_length=400,
        win_length=1600)
    spectrogram = np.abs(spectrogram)
    spectrogram = np.dot(
        librosa.filters.mel(sr, 2048, n_mels=80, fmin=0, fmax=8000),
        spectrogram)
    spectrogram = np.log(spectrogram*SCALING + 1e-2)
    return np.transpose(spectrogram)


def run_wavernn(model, spectrogram, output):
    """
    Run inference using a frozen model.

    Args:
        model:       Frozen graph file, .pb format.
        spectrogram: 80-band mel spectrogram.
        output:      Output file.

    Returns:
        Output audio, 16 kHz sample rate.
    """
    # Pad the spectrograms (in the time dimension) before input.
    padding = 12
    spectrogram = np.pad(spectrogram, [[padding, padding], [0, 0]],
                         mode='constant')

    with tf.gfile.GFile(model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:
            tf.import_graph_def(graph_def, name="")
            print("Generating samples...")
            start_time = time.time()

            writer = tf.summary.FileWriter("./traning_graph")
            writer.add_graph(session.graph)

            audio, residual, upsampling, \
            weights_W, biases_W, weights_R, biases_R, weights_R_W, biases_R_W, weights_O_W, biases_O_W, \
            weights_A_E, biases_A_E, \
            moving_Var_1, moving_Mean_1, gamma_1, beta_1, weights_1, biases_1, \
            moving_Var_2, moving_Mean_2, gamma_2, beta_2, weights_2, biases_2, \
            moving_Var_3, moving_Mean_3, gamma_3, beta_3, weights_3, biases_3 \
            = session.run([OUTPUT_NODE, Residual, Upsampleing, \
                           Weights_W, Biases_W, Weights_R, Biases_R, Weights_R_W, Biases_R_W, Weights_O_W, Biases_O_W, \
                           Weights_A_E, Biases_A_E, \
                           Moving_Var_1, Moving_Mean_1, Gamma_1, Beta_1, Weights_1, Biases_1, \
                           Moving_Var_2, Moving_Mean_2, Gamma_2, Beta_2, Weights_2, Biases_2, \
                           Moving_Var_3, Moving_Mean_3, Gamma_3, Beta_3, Weights_3, Biases_3 \
                          ], 
                          feed_dict={
                              INPUT_NODE: [spectrogram],
                              TRAINING: True,
                          })
            elapsed = time.time() - start_time
            generated_seconds = audio.size / SAMPLE_RATE
            path = './output_parameters.npz'
            np.savez(path, \
                     weights_W=weights_W, biases_W=biases_W, weights_R=weights_R, biases_R=biases_R, weights_R_W=weights_R_W, biases_R_W=biases_R_W, weights_O_W=weights_O_W, biases_O_W=biases_O_W, \
                     weights_A_E=weights_A_E, biases_A_E=biases_A_E, \
                     moving_Var_1=moving_Var_1, moving_Mean_1=moving_Mean_1, gamma_1=gamma_1, beta_1=beta_1, weights_1=weights_1, biases_1=biases_1, \
                     moving_Var_2=moving_Var_2, moving_Mean_2=moving_Mean_2, gamma_2=gamma_2, beta_2=beta_2, weights_2=weights_2, biases_2=biases_2, \
                     moving_Var_3=moving_Var_3, moving_Mean_3=moving_Mean_3, gamma_3=gamma_3, beta_3=beta_3, weights_3=weights_3, biases_3=biases_3)              

    print("Generated {:.2f}s in {:.2f}s ({:.3f}x realtime)."
        .format(generated_seconds, elapsed, generated_seconds / elapsed))
    return audio


def plot_spectrogram(spectrogram):
    librosa.display.specshow(np.transpose(spectrogram), cmap="plasma")
    plt.tight_layout()
    plt.savefig("spectrogram.png", bbox_inches=None, pad_inches=0)
    plt.close()


if __name__ == '__main__':
    inference()
