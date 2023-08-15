import numpy as np
import scipy.io as sio
from tqdm import tqdm

seed_value = 42

def fake_data_task1(subject, task_type):
    np.random.seed(seed_value)

    mean = 0  # Mean of the Gaussian distribution
    std_dev = 1  # Standard deviation of the Gaussian distribution

    word_fields = (
    'FFD_t1',
    'FFD_t2',
    'FFD_a1',
    'FFD_a2',
    'FFD_b1',
    'FFD_b2',
    'FFD_g1',
    'FFD_g2',

    'TRT_t1',
    'TRT_t2',
    'TRT_a1',
    'TRT_a2',
    'TRT_b1',
    'TRT_b2',
    'TRT_g1',
    'TRT_g2',

    'GD_t1',
    'GD_t2',
    'GD_a1',
    'GD_a2',
    'GD_b1',
    'GD_b2',
    'GD_g1',
    'GD_g2')


    if task_type == "task1":
        fields = ('mean_t1',
        'mean_t2',
        'mean_a1',
        'mean_a2',
        'mean_b1',
        'mean_b2',
        'mean_g1',
        'mean_g2',
        'answer_mean_t1',
        'answer_mean_t2',
        'answer_mean_a1',
        'answer_mean_a2',
        'answer_mean_b1',
        'answer_mean_b2',
        'answer_mean_g1',
        'answer_mean_g2')
    elif task_type == "task2" or task_type == "task3":
        fields = ('mean_t1',
        'mean_t2',
        'mean_a1',
        'mean_a2',
        'mean_b1',
        'mean_b2',
        'mean_g1',
        'mean_g2')

    file_size = subject['sentenceData'].shape[1]
    for i in range(file_size):
        if np.isnan(subject['sentenceData'][0,i]["rawData"][0,0]):
            continue
    for field in fields:
        size = subject['sentenceData'][0,i][field].shape
        subject['sentenceData'][0,i][field] = np.random.normal(mean, std_dev, size)

    for i in range(file_size):
        word_size = subject['sentenceData'][0,i]["word"].shape[1]
        for w in range(word_size):
            for word_field in word_fields:
                try: #some files do not have word_field because it is empty. Try to see if it has word_field
                    size = subject['sentenceData'][0,i]["word"][0,w][word_field].shape
                except:#if it does not have word_field (which means empty), then skip
                    size = -1
                if 'FFD' in word_field and size != -1:
                    if subject['sentenceData'][0,i]["word"][0,w]['FFD'].shape[0] == 0: # if it is na, then skip
                        continue
                    else:
                        subject['sentenceData'][0,i]["word"][0,w][word_field] = np.random.normal(mean, std_dev, size)
                if 'TRT' in word_field and size != -1:
                    if subject['sentenceData'][0,i]["word"][0,w]['TRT'].shape[0] == 0: # if it is na, then skip
                        continue
                    else:
                        subject['sentenceData'][0,i]["word"][0,w][word_field] = np.random.normal(mean, std_dev, size)
                if 'GD' in word_field and size != -1:
                    if subject['sentenceData'][0,i]["word"][0,w]['GD'].shape[0] == 0: # if it is na, then skip
                        continue
                    else:
                        subject['sentenceData'][0,i]["word"][0,w][word_field] = np.random.normal(mean, std_dev, size)
    return subject

if __name__ == "__main__":
    # Load the mat file in task 1
    subject_names = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH']
    for subject_name in tqdm(subject_names):
        print("******* Start processing subject = ", subject_name, " *******")

        task_type = "task1"
        subject = sio.loadmat(f'dataset/ZuCo/task1-SR/Matlab_files/results{subject_name}_SR.mat')
        fake_subject = fake_data_task1(subject, task_type)
        sio.savemat(f'dataset/ZuCo/task1-SR/Matlab_files/results{subject_name}_SR.mat', fake_subject)

        task_type = "task2"
        subject = sio.loadmat(f'dataset/ZuCo/task2-NR/Matlab_files/results{subject_name}_NR.mat')
        fake_subject = fake_data_task1(subject, task_type)
        sio.savemat(f'dataset/ZuCo/task2-NR/Matlab_files/results{subject_name}_NR.mat', fake_subject)

        task_type = "task3"
        subject = sio.loadmat(f'dataset/ZuCo/task3-TSR/Matlab_files/results{subject_name}_TSR.mat')
        fake_subject = fake_data_task1(subject, task_type)
        sio.savemat(f'dataset/ZuCo/task3-TSR/Matlab_files/results{subject_name}_TSR.mat', fake_subject)
