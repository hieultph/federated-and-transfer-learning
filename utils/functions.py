from sklearn.utils import shuffle
import os
import pandas as pd

def get_df_from_dir(dir):
  data_paths = []
  data_labels = []

  for label in os.listdir(dir):
      for image in os.listdir(dir+label):
          data_paths.append(dir+label+'/'+image)
          data_labels.append(label)

  data_paths, data_labels = shuffle(data_paths, data_labels)

  Fseries = pd.Series(data_paths, name= 'filepaths')
  Lseries = pd.Series(data_labels, name='labels')

  return pd.concat([Fseries, Lseries], axis= 1)

def generate_clients_data(number_of_clients, df):

    number_of_samples = len(df['filepaths'])
    number_of_samples_per_client = number_of_samples // number_of_clients

    clients_dataset = {}

    # Shuffle the DataFrame to ensure random distribution of data
    df = df.sample(frac=1).reset_index(drop=True)

    # Calculate total samples needed
    total_samples = number_of_clients * number_of_samples_per_client
    if total_samples > len(df):
        raise ValueError("There are not enough samples to distribute among clients.")

    # Ensure each client gets the specified number of samples
    for client_id in range(number_of_clients):
        start_index = client_id * number_of_samples_per_client
        end_index = start_index + number_of_samples_per_client
        client_data = df.iloc[start_index:end_index]

        # Assigning each client their slice of data
        clients_dataset.update({client_id: client_data})

    return clients_dataset

