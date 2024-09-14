# -*- coding: utf-8 -*-
"""EcgGAN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bn4CVtMpknBVn6saw4A8eYbyipa3zQZl
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline

!unzip ecg-id-database-1.0.0.zip

general_directory = os.path.join("/content/ecg-id-database-1.0.0")
enteries = os.listdir(general_directory)

fs = 500  # Sampling frequency indicated by the publisher of the database

# Initializing lists to store extracted features
subject, age, gender, RR, ecg_mean, ecg_std, ecg_var, ecg_median, ecg_samples = [], [], [], [], [], [], [], [], []

# Loop through each folder to extract the relevant information
for folder in enteries:
    if "Person_" in folder:
        info_targetPattern = os.path.join(general_directory, folder, '*.hea')
        info_files = glob.glob(info_targetPattern)

        for ecg_info in info_files:
            try:
                with open(ecg_info) as f:
                    subject_info = int(folder.replace("Person_", "", 1))

                    # Extracting demographic information
                    lines = [line.rstrip() for line in f]
                    age_info = int([int(s) for s in lines[4].split() if s.isdigit()][0])
                    condition = lines[5].find('female')
                    gender_info = 'female' if condition > 0 else 'male'

                    ecg_signal_file = ecg_info.replace(".hea", ".dat")
                    ecg_signal = np.fromfile(ecg_signal_file, dtype='int32')
                    ecg_signal = ecg_signal / np.max(np.abs(ecg_signal))  # Normalizing signal
                    ts = np.arange(0, len(ecg_signal) / fs, 1 / fs)

                    # RR intervals calculation
                    pks_RR = sp.signal.find_peaks(ecg_signal, height=np.mean(np.abs(ecg_signal) * 5), distance=500)[0]

                    RR_samples = np.diff(pks_RR)
                    RR_time = RR_samples * ts[1]

                    # Handling cases where not enough RR intervals are found
                    if len(pks_RR) <= 1:
                        subject.append(subject_info)
                        age.append(age_info)
                        gender.append(gender_info)
                        ecg_mean.append(np.nan)
                        ecg_std.append(np.nan)
                        ecg_var.append(np.nan)
                        ecg_median.append(np.nan)
                        RR.append(np.nan)
                        ecg_samples.append(np.nan)
                    else:
                        for index in range(len(pks_RR) - 1):
                            subject.append(subject_info)
                            age.append(age_info)
                            gender.append(gender_info)
                            ecg_mean.append(np.mean(ecg_signal[pks_RR[index]:pks_RR[index + 1]]))
                            ecg_std.append(np.std(ecg_signal[pks_RR[index]:pks_RR[index + 1]]))
                            ecg_var.append(np.var(ecg_signal[pks_RR[index]:pks_RR[index + 1]]))
                            ecg_median.append(np.median(ecg_signal[pks_RR[index]:pks_RR[index + 1]]))
                            RR.append(RR_time[index])
                            ecg_samples.append(ecg_signal[pks_RR[index]:pks_RR[index + 1]])
            except Exception as e:
                print(f"Error processing file {ecg_info}: {e}")

# Creating the DataFrame
df = pd.DataFrame({
    'Subject': subject,
    'Age': age,
    'Gender': gender,
    'RR_Interval': RR,
    'ECG_Mean': ecg_mean,
    'ECG_Std': ecg_std,
    'ECG_Var': ecg_var,
    'ECG_Median': ecg_median,
    'ECG_Samples': ecg_samples
})

# Convert columns to appropriate numeric types
df['RR_Interval'] = pd.to_numeric(df['RR_Interval'], errors='coerce')
df['ECG_Mean'] = pd.to_numeric(df['ECG_Mean'], errors='coerce')
df['ECG_Std'] = pd.to_numeric(df['ECG_Std'], errors='coerce')
df['ECG_Var'] = pd.to_numeric(df['ECG_Var'], errors='coerce')
df['ECG_Median'] = pd.to_numeric(df['ECG_Median'], errors='coerce')

df.dropna(inplace=True)
df = df.sort_values(by=['Subject'])
df.reset_index(inplace=True, drop=True)
df.head()

signal_lenghts = [i.__len__() for i in df['ECG_Samples'].values[0:]]
print(max(signal_lenghts)), print(np.mean(signal_lenghts)), print(min(signal_lenghts))

plt.plot(df['ECG_Samples'][0])
plt.title('Distribution of one signal in dataset');
plt.show();

# Show a preview of the dataframe
df.info()

# Step 1: Get unique subject IDs in the dataset
unique_subjects = sorted(df['Subject'].unique())

# Step 2: Create a mapping from old subject IDs to new consecutive values
subject_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_subjects, start=1)}

# Step 3: Apply the mapping to the 'Subject' column
df['Subject'] = df['Subject'].map(subject_mapping)

plt.figure(figsize=(15, 7))
sns.countplot(df, x='Subject', hue='Subject', palette=sns.color_palette("crest", as_cmap=True))
plt.title("Number of samples for each subject");
plt.xticks(rotation=90);
plt.show();

# find signals lenght and get the longest lenght of them,
# all other signals should be padded into the max lenght or mean lenght in order
# to have integrated length
signal_lenghts = [i.__len__() for i in df['ECG_Samples'].values[0:]]
max_length, mean_lenght = max(signal_lenghts), round(np.mean(signal_lenghts))  # Maximum & Mean length of ECG samples
print(max_length), print(mean_lenght)

# Define a fixed signal length (based on the max length in your dataset)
max_signal_length = max_length  # Example value, adjust accordingly

# Function to pad or truncate ECG signals
def pad_truncate_signal_max(signal, target_length=max_length):
    if len(signal) > target_length:
        return signal[:target_length]
    elif len(signal) < target_length:
        return np.pad(signal, (0, target_length - len(signal)), 'constant')
    else:
        return signal

# Function to pad or truncate ECG signals with mean padding
def pad_truncate_signal_mean(signal, target_length=mean_lenght):
    signal_mean = np.mean(signal)  # Calculate the mean of the signal

    if len(signal) > target_length:
        # Truncate the signal if it's longer than the target length
        return signal[:target_length]
    elif len(signal) < target_length:
        # Pad the signal with its mean value if it's shorter than the target length
        padding = (0, target_length - len(signal))
        return np.pad(signal, padding, 'constant', constant_values=(signal_mean,))
    else:
        return signal  # Return the original signal if it's the right length

# Function to apply moving average smoothing
def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

# Apply the function to each ECG sample in the DataFrame
df['ECG_Samples'] = df['ECG_Samples'].apply(lambda x: pad_truncate_signal_mean(x))

# Apply the smoothing function to each ECG sample in the DataFrame
df['Smoothed_ECG_Samples'] = df['ECG_Samples'].apply(lambda x: moving_average(np.array(x), window_size=15))

plt.subplot(2, 1, 1)
plt.plot(df['ECG_Samples'][0])
plt.title('padded signal before smoothing')
plt.subplot(2, 1, 2)
plt.plot(df['Smoothed_ECG_Samples'][0])
plt.title('Smoothed signal')
plt.subplots_adjust(hspace=0.5)
plt.show()

"""Huber Loss – Loss function to use in Regression when dealing with Outliers"""

# Define the dimensions
input_dim = len(df['Smoothed_ECG_Samples'][0])  # The original signal length, e.g., 6861
encoding_dim = 50  # Reduced dimensionality, e.g., 50

# Ensure that your 'Smoothed_ECG_Samples' is a NumPy array with correct shape (n_samples, input_dim)
# Convert 'Smoothed_ECG_Samples' from the DataFrame to a NumPy array
ecg_samples_np = np.array(df['Smoothed_ECG_Samples'].tolist())

# Check the shape of the data to confirm it's (n_samples, `max_lenght or mean lenght`)
print(ecg_samples_np.shape)  # Should print something like (2832, `max_lenght or mean lenght`)

# Define the encoder model
input_signal = Input(shape=(input_dim,))
encoded = Dense(128, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform(), kernel_regularizer=l2(0.001))(input_signal)
encoded = Dropout(0.2)(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(256, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform(), kernel_regularizer=l2(0.001))(encoded)
encoded = Dropout(0.2)(encoded)
encoded = BatchNormalization()(encoded)

encoded = Dense(encoding_dim, activation='tanh')(encoded)

# Define the decoder model
decoded = Dense(128, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform(), kernel_regularizer=l2(0.001))(encoded)
decoded = Dropout(0.2)(decoded)
decoded = BatchNormalization()(decoded)
decoded = Dense(256, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform(), kernel_regularizer=l2(0.001))(decoded)
decoded = Dropout(0.2)(decoded)
decoded = BatchNormalization()(decoded)
decoded = Dense(input_dim, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform())(decoded)

# Define the full autoencoder model
autoencoder = Model(input_signal, decoded)

# Decrease the learning rate
optimizer = Adam(learning_rate=0.001)

early_stopping = EarlyStopping(monitor='loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)

# Compile the autoencoder
autoencoder.compile(optimizer=optimizer, loss='huber')

# Train the autoencoder
autoencoder.fit(ecg_samples_np, ecg_samples_np, epochs=100, batch_size=64, shuffle=True, callbacks=[early_stopping, reduce_lr])

# Extract the encoder model to get the compressed representations
encoder = Model(input_signal, encoded)
decoder = Model(encoded, decoded)

# Now you can encode and decode the data
compressed_signals = encoder.predict(ecg_samples_np)  # Encode the data
reconstructed_signals = decoder.predict(compressed_signals)  # Decode the data back to original

plt.subplot(3, 1, 1)
plt.plot(compressed_signals[0])
plt.title('Compressed signal with encoder')
plt.subplot(3, 1, 2)
plt.plot(reconstructed_signals[0])
plt.title('Reconstructed signal with decoder')
plt.subplot(3, 1, 3)
plt.plot(ecg_samples_np[0])
plt.title('Actual signal')
plt.subplots_adjust(hspace=0.5)
plt.show()

# create a list contain all numpy arrays of decomposited signals and add it to original dataframe
compressed_list = [np.array(i) for i in compressed_signals]
compressed_signals_df = pd.DataFrame({'compressed_signals': compressed_list})
concat_df = pd.merge(df, compressed_signals_df, left_index=True, right_index=True)
print(concat_df.head())

!pip install -q git+https://github.com/tensorflow/docs

import keras
from keras import layers
from keras import ops
from tensorflow_docs.vis import embed
import tensorflow as tf
import numpy as np
import imageio

# Prepare dataset: decomposed signals and one-hot encoded subject labels
decomposed_signals = np.array(concat_df['compressed_signals'].tolist())
subject_labels = pd.get_dummies(concat_df['Subject']).values  # One-hot encode subject labels

# Combine decomposed signals and subject labels in a TensorFlow Dataset
batch_size = 64
latent_dim = 128
num_subjects = subject_labels.shape[1]
signal_dim = decomposed_signals.shape[1]

dataset = tf.data.Dataset.from_tensor_slices((decomposed_signals, subject_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

# Label embedding size
label_embedding_size = 50

# Generator with label embedding
generator_input = keras.Input(shape=(latent_dim,))
label_input = keras.Input(shape=(num_subjects,))
label_embedding = keras.layers.Dense(label_embedding_size)(label_input)
x = keras.layers.Concatenate()([generator_input, label_embedding])
x = layers.Dense(128)(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)
x = layers.Dense(256)(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)
generator_output = layers.Dense(signal_dim, activation="tanh")(x)

generator = keras.Model([generator_input, label_input], generator_output, name="generator")

# Discriminator model with label embedding
discriminator_input = keras.Input(shape=(signal_dim,))
label_input_disc = keras.Input(shape=(num_subjects,))
label_embedding_disc = keras.layers.Dense(label_embedding_size)(label_input_disc)
x_disc = keras.layers.Concatenate()([discriminator_input, label_embedding_disc])
x_disc = layers.Dense(256)(x_disc)
x_disc = layers.LeakyReLU(negative_slope=0.2)(x_disc)
x_disc = layers.Dense(128)(x_disc)
x_disc = layers.LeakyReLU(negative_slope=0.2)(x_disc)
discriminator_output = layers.Dense(1)(x_disc)

discriminator = keras.Model([discriminator_input, label_input_disc], discriminator_output, name="discriminator")


class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator(1337)
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.feature_matching_loss_tracker = keras.metrics.Mean(name="feature_matching_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker, self.feature_matching_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        real_signals, one_hot_labels = data
        batch_size = tf.shape(real_signals)[0]

        # Generate random latent vectors and concatenate with labels
        random_latent_vectors = keras.random.normal(shape=(batch_size, self.latent_dim), seed=self.seed_generator)
        generated_signals = self.generator([random_latent_vectors, one_hot_labels])

        # Combine real and fake signals for the discriminator
        fake_signal_and_labels = [generated_signals, one_hot_labels]
        real_signal_and_labels = [real_signals, one_hot_labels]
        combined_signals = [ops.concatenate([real_signals, generated_signals], axis=0),
                            ops.concatenate([one_hot_labels, one_hot_labels], axis=0)]

        # Labels for real (1) and fake (0) signals
        labels = ops.concatenate([ops.ones((batch_size, 1)), ops.zeros((batch_size, 1))], axis=0)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_signals)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Sample new random latent vectors for generator training
        random_latent_vectors = keras.random.normal(shape=(batch_size, self.latent_dim), seed=self.seed_generator)

        # Create labels for the generator: "all real"
        misleading_labels = ops.ones((batch_size, 1))

        # Train the generator (without updating the discriminator)
        with tf.GradientTape() as tape:
            fake_signals = self.generator([random_latent_vectors, one_hot_labels])
            fake_signal_and_labels = [fake_signals, one_hot_labels]
            predictions = self.discriminator(fake_signal_and_labels)

            # Feature matching loss (L2 distance between real and fake feature maps in the discriminator)
            real_features = self.discriminator(real_signal_and_labels, training=False)
            fake_features = self.discriminator(fake_signal_and_labels, training=False)
            feature_matching_loss = tf.reduce_mean(tf.square(real_features - fake_features))

            # Combine generator loss and feature matching loss
            g_loss = self.loss_fn(misleading_labels, predictions) + feature_matching_loss
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        self.feature_matching_loss_tracker.update_state(feature_matching_loss)

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            "feature_matching_loss": self.feature_matching_loss_tracker.result(),
        }

# Visualization of generated signals
def plot_generated_signals(generator, epoch, num_samples=5):
    random_latent_vectors = keras.random.normal(shape=(num_samples, latent_dim))
    random_labels = np.eye(num_subjects)[np.random.choice(num_subjects, num_samples)]
    generated_signals = generator.predict([random_latent_vectors, random_labels])

    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.plot(generated_signals[i])
        plt.title(f"Generated Signal (Label {np.argmax(random_labels[i])})")
    plt.show()

# Callbacks for early stopping and model checkpointing
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "gan_checkpoint.weights.h5", monitor="g_loss", save_best_only=True, save_weights_only=True
)

early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor="g_loss", patience=20, restore_best_weights=True, mode='min'
)

# Compile and train the model
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

# Custom training loop to include signal visualization
epochs = 1000
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    cond_gan.fit(dataset, epochs=1, callbacks=[checkpoint_callback, early_stopping_callback])

    # Every 50 epochs, visualize the generated signals
    if (epoch + 1) % 50 == 0:
        plot_generated_signals(generator, epoch)

def generate_ecg_for_subject(subject_id, num_signals=1):
    """
    Generate and reconstruct ECG signals for a given subject using the trained CGAN model.

    Parameters:
    - subject_id (int): The ID of the subject for which to generate signals.
    - num_signals (int): Number of signals to generate.

    Returns:
    - generated_ecg_signals (list of np.ndarray): List of generated ECG signals for the subject.
    """
    # Convert subject ID to one-hot encoding
    subject_one_hot = np.zeros((num_signals, num_subjects))
    subject_one_hot[:, subject_id - 1] = 1  # subject_id should be between 1 and num_subjects

    # Ensure both latent vectors and subject_one_hot are tensors
    subject_one_hot_tensor = tf.convert_to_tensor(subject_one_hot, dtype=tf.float32)

    # Generate random latent vectors as tensors
    random_latent_vectors = keras.random.normal(shape=(num_signals, latent_dim))

    # Generate decomposed signals using the generator (pass tensors directly)
    generated_decomposed_signals = generator([random_latent_vectors, subject_one_hot_tensor]).numpy()

    # Reconstruct the original ECG signals using PCA inverse transformation (decoder)
    generated_ecg_signals = decoder.predict(generated_decomposed_signals)

    return generated_ecg_signals


def plot_generated_signals_for_subjects(subject_ids, num_signals=1):
    """
    Generate and plot ECG signals for a list of Subject IDs.

    Parameters:
    - subject_ids (list of int): List of subject IDs to generate and compare signals for.
    - num_signals (int): Number of signals to generate per subject.
    """
    plt.figure(figsize=(10, len(subject_ids) * 4))

    for i, subject_id in enumerate(subject_ids):
        # Ensure subject_id is in range (1-based index)
        subject_id = max(1, subject_id)

        # Generate signals for the current subject
        generated_signals = generate_ecg_for_subject(subject_id, num_signals=num_signals)

        # Plot the generated signals
        plt.subplot(len(subject_ids), 1, i + 1)
        plt.plot(generated_signals[0], label=f'Generated Signal (Subject {subject_id})', alpha=0.6)

        # Add title and labels
        plt.title(f'Generated ECG Signal for Subject {subject_id}')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.legend()

    plt.tight_layout()
    plt.show()


# Generate and compare signals for Subject IDs 1, 2, and 3
plot_generated_signals_for_subjects(subject_ids=[1, 2, 3], num_signals=1)

def generate_signals_for_all_records(df, num_signals=1):
    """
    Generate synthetic signals for each record in the DataFrame and store them in a new column 'generated_signals'.

    Parameters:
    - df: The original DataFrame (concat_df) containing the decomposed signals.
    - num_signals: The number of signals to generate per record (default is 1).

    Returns:
    - df: Updated DataFrame with a new column 'generated_signals'.
    """
    generated_signals_list = []

    for i, row in df.iterrows():
        subject_id = row['Subject']  # Fetch the subject ID
        # Generate signals using GAN for this subject
        generated_signal = generate_ecg_for_subject(subject_id, num_signals=num_signals)[0]
        # Denoise or threshold if needed (already part of generate_ecg_for_subject if necessary)
        # generated_signal = denoise_signal(generated_signal)
        generated_signals_list.append(generated_signal)

    # Add the generated signals to the DataFrame
    df['generated_signals'] = generated_signals_list
    return df

# Update the concat_df DataFrame with generated signals
concat_df_update = generate_signals_for_all_records(concat_df, num_signals=1)

# Suppose concat_df_update is your DataFrame with generated signals
# Create a DataFrame for generated signals
generated_signals_df = concat_df_update[['generated_signals']].copy()
generated_signals_df['label'] = 1  # Label for generated signals

# Create a DataFrame for original ECG signals
original_signals_df = concat_df_update[['Smoothed_ECG_Samples']].copy()
original_signals_df['label'] = 0  # Label for original signals

# Check lengths
print(f"Length of generated signals DataFrame: {len(generated_signals_df)}")
print(f"Length of original signals DataFrame: {len(original_signals_df)}")

# Ensure signals and labels have the same length
signals = generated_signals_df['generated_signals'].tolist() + original_signals_df['Smoothed_ECG_Samples'].tolist()
labels = np.concatenate([
    np.ones(len(generated_signals_df)),  # Labels for generated signals
    np.zeros(len(original_signals_df))   # Labels for original signals
])

print(f"Length of signals: {len(signals)}")
print(f"Length of labels: {len(labels)}")


def extract_features(signals):
    features = []
    for signal in signals:
        mean = np.mean(signal)
        variance = np.var(signal)
        # Add more features if needed
        features.append([mean, variance])
    return np.array(features)

# Extract features
X_features = extract_features(signals)

print(f"Shape of X_features: {X_features.shape}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, labels, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_test_scaled)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Identify indices of generated signals misclassified as real
misclassified_indices = np.where((y_pred == 1) & (y_test == 0))[0]

print(f"Number of generated signals misclassified as real: {len(misclassified_indices)}")

misclassified_indices[:5]

# Extract the misclassified signals from the test set
misclassified_signals = [signals[i] for i in misclassified_indices]

# Display some of the misclassified signals
num_to_show = min(1, len(misclassified_signals))  # Number of signals to display
plt.figure(figsize=(10, num_to_show * 2))

for i in range(num_to_show):
    plt.subplot(num_to_show, 1, i + 1)
    plt.plot(misclassified_signals[i])
    plt.title(f'Misclassified Generated Signal as Real (Index {misclassified_indices[i]})')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Get the signals classified as "real" by the SVM model
real_signals_idx = np.where(y_pred == 0)[0]  # Indices of signals classified as real

# Filter those signals from the combined list
real_signals = [signals[i] for i in real_signals_idx]

# If the original DataFrame contains subject IDs, we can also filter the corresponding IDs
real_subject_ids = concat_df_update['Subject'].tolist()
real_subject_ids = [real_subject_ids[i % len(concat_df_update)] for i in real_signals_idx]

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Conv1D, MaxPooling1D, Dense, Flatten, Dropout

def build_light_lstm_cnn(input_shape, num_subjects):
    inputs = Input(shape=input_shape)

    # LSTM layer
    lstm_out = LSTM(32, return_sequences=True)(inputs)

    # CNN layers
    cnn_out = Conv1D(32, kernel_size=3, activation='relu')(lstm_out)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)

    # Flatten for dense layers
    flat_out = Flatten()(cnn_out)

    # Dense layers
    dense_out = Dense(64, activation='relu')(flat_out)
    dense_out = Dropout(0.3)(dense_out)

    # Output layer for multi-class classification (subject IDs)
    outputs = Dense(num_subjects, activation='softmax')(dense_out)

    model = Model(inputs, outputs)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Assuming input_shape based on length of signals and number of subjects
input_shape = (len(real_signals[0]), 1)  # Example for 1D ECG signals
num_subjects = len(concat_df_update['Subject'].unique())  # Number of unique subjects

# Build the model
subject_classification_model = build_light_lstm_cnn(input_shape, num_subjects)
subject_classification_model.summary()

# Reshape the signals to 3D for LSTM-CNN model (samples, timesteps, features)
real_signals_array = np.array(real_signals)
real_signals_array = real_signals_array.reshape((real_signals_array.shape[0], real_signals_array.shape[1], 1))

# Convert subject IDs to a numeric format
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
real_subject_ids_encoded = label_encoder.fit_transform(real_subject_ids)

from sklearn.model_selection import train_test_split

# Split the real signals into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(real_signals_array, real_subject_ids_encoded, test_size=0.2, random_state=42)

# Train the LSTM-CNN model
subject_classification_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = subject_classification_model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")

# Get predicted probabilities for each instance in X_test
y_pred_probs = subject_classification_model.predict(X_test)

# Get the index of the highest probability for each sample, which corresponds to the predicted class (subject ID)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Convert the predicted class indices back to the original subject IDs using the label encoder
y_pred_subject_ids = label_encoder.inverse_transform(y_pred_classes)

# Print a few predictions along with their corresponding true subject IDs
for i in range(10):  # Display the first 10 predictions
    print(f"Predicted Subject ID: {y_pred_subject_ids[i]}, True Subject ID: {label_encoder.inverse_transform([y_test[i]])[0]}")
