import glob
import shutil
import os
import cv2
from typing import Iterator, Tuple, Any
from scipy.spatial.transform import Rotation

import glob
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
import numpy as np
import tensorflow_datasets as tfds
# import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def _parse_example(episode_path, goal_dataset,embed=None):
    data = {}
    # leader_path = os.path.join(episode_path, 'leader/*.pt')
    # follower_path = os.path.join(episode_path, 'follower/*.pt')
    path = os.path.join(episode_path,'*.pt')
    #path = os.path.join(episode_path, "*.pickle")
    for file in glob.glob(path):

        # Keys contained in .pickle:
        # 'joint_state', 'joint_state_velocity', 'des_joint_state', 'des_joint_vel', 'end_effector_pos', 'end_effector_ori', 'des_gripper_width', 'delta_joint_state',
        # 'delta_des_joint_state', 'delta_end_effector_pos', 'delta_end_effector_ori', 'language_description', 'traj_length'
        #pt_file_path = os.path.join(episode_path, file)
        name = Path(file).stem
        data.update({name : torch.load(file)})
    # for file in glob.glob(episode_path):
    #     name = 'des_' + Path(file).stem
    #     data.update({name : torch.load(file)})

    trajectory_length = len(data[list(data.keys())[0]])
    
    for feature in list(data.keys()):
        for i in range(len(data[feature])):
            data[f'delta_{feature}'] = torch.zeros_like(torch.tensor(data[feature]))
            if i == 0:
                data[f'delta_{feature}'][i] = 0
            else:
                data[f'delta_{feature}'][i] = data[feature][i] - data[feature][i-1]






  
    top_cam_path = os.path.join(episode_path, 'images/overhead_cam_orig')
    wrist_left_cam_path = os.path.join(episode_path, 'images/wrist_cam_left_orig')
    wrist_right_cam_path = os.path.join(episode_path, 'images/wrist_cam_right_orig')
    # top_cam_path = os.path.join(episode_path, 'images/cam_high_orig')
    # wrist_left_cam_path = os.path.join(episode_path, 'images/cam_left_wrist_orig')
    # wrist_right_cam_path = os.path.join(episode_path, 'images/cam_right_wrist_orig')
    top_cam_vector = create_img_vector(top_cam_path, trajectory_length)
    wrist_left_cam_vector = create_img_vector(wrist_left_cam_path, trajectory_length)
    wrist_right_cam_vector = create_img_vector(wrist_right_cam_path, trajectory_length)
    # cam1_image_vector = create_img_vector(cam1_path, trajectory_length)
    # cam2_image_vector = create_img_vector(cam2_path, trajectory_length)
    data.update({
                'image_top': top_cam_vector, 
                'image_wrist_left' : wrist_left_cam_vector, 
                'image_wrist_right' : wrist_right_cam_vector
                })




    for i in range(trajectory_length):
        # compute Kona language embedding
        #language_embedding = embed(data['language_description']).numpy() if embed is not None else [np.zeros(512)]
        # action = np.append(data['delta_end_effector_pos'][i], delta_quat.as_euler("xyz"), axis=0)
        # action = np.append(action, data['des_gripper_width'][i])
        # action_abs = np.append(data['des_end_effector_pos'][i], abs_quat.as_euler("xyz"), axis=0)
        # action_abs = np.append(action_abs, data['des_gripper_width'][i])
        # action = data['delta_ee_pos'][i]
        # action = np.append(action, data['des_gripper_state'][i])
        # action_abs = data['des_ee_pos'][i]
        # action_abs = np.append(action_abs, data['des_gripper_state'][i])
        # action = data['des_joint_state'][i]
        action_all_joint = torch.zeros(14)
        observation_all_joint = torch.zeros(14)

        action_all_joint = np.float32(data['leader_joint_pos'][i])
    


        observation_all_joint = np.float32(data['follower_joint_pos'][i])

        goal_dataset.add_frame(
           {
                "observation.images.overhead_cam":  data['image_top'][i],
        "observation.images.wrist_cam_left":data['image_wrist_left'][i],
        "observation.images.wrist_cam_right": data['image_wrist_right'][i],
        # "observation.state": observation_all_joint,
        "action": action_all_joint,
        "task":"cube_transfer"
           }
       )
    goal_dataset.save_episode()

def create_img_vector(img_folder_path, trajectory_length):
    cam_list = []
  
    img_paths = glob.glob(os.path.join(img_folder_path, '*.jpg'))
    img_paths = sorted(img_paths, key=lambda x: float(Path(x).stem))
   
    assert len(img_paths)==trajectory_length, "Number of images does not equal trajectory length!"

    for img_path in img_paths:
        #img_array = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
        img_array = cv2.imread(img_path)
        cam_list.append(img_array)
    return cam_list


def get_trajectorie_paths_recursive(directory, sub_dir_list):
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            sub_dir_list.append(directory) if entry == "images" else get_trajectorie_paths_recursive(full_path, sub_dir_list)

if __name__ == "__main__":
    data_path = "/home/simon/collections/Simulation/cube_transfer_right_2_left_50" 
    #embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    # create list of all examples
    repo_name = "simon/aloha_cube_transfer_test"
    raw_dirs = []
    get_trajectorie_paths_recursive(data_path, raw_dirs)



    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    goal_dataset = LeRobotDataset.create(
    repo_id= repo_name,
    robot_type="aloha",
    fps=50,
    features={
        "observation.images.overhead_cam": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist_cam_left": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist_cam_right": {
            "dtype": "image",
            "shape": (224,224, 3),
            "names": ["height", "width", "channel"],
        },
        # "observation.state": {
        #     "dtype": "float32",
        #     "shape": (14,),
        #     "names": ["state"],
        # },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["actions"],
        },
    },
    image_writer_threads=10,
    image_writer_processes=5,
)


    for trajectorie_path in tqdm(raw_dirs):
        _parse_example(trajectorie_path, goal_dataset)
        #print(sample)