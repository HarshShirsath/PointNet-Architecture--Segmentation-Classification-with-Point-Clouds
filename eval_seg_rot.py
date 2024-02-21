import numpy as np
import argparse
import os
import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg_rot
from scipy.spatial.transform import Rotation

def rotate_point_cloud(point_cloud, angle_degrees):
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Create a rotation matrix around the z-axis
    rotation_matrix = Rotation.from_euler('z', angle_radians).as_matrix()

    # Apply rotation to the point cloud
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix.T)

    # Convert to torch tensor and ensure the data type is float32
    rotated_point_cloud = torch.from_numpy(rotated_point_cloud).float()

    return rotated_point_cloud

def create_parser():
    """Creates a parser for command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of samples in a batch')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="seg", help='The task: cls or seg')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # Initialize Model for Segmentation Task
    model = seg_model().to(args.device)

    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("Successfully loaded checkpoint from {}".format(model_path))

    # Initialize Test Data Loader for Segmentation Task
    test_dataloader = get_data_loader(args=args, train=False)

    # Rotate input point clouds by 30 degrees
    rotation_angle = 30

    correct_point = 0
    num_point = 0
    preds_labels = []

    for batch in test_dataloader:
        point_clouds, labels = batch
        rotated_point_clouds = rotate_point_cloud(point_clouds, rotation_angle).to(args.device)
        labels = labels.to(args.device).to(torch.long)

        with torch.no_grad():
            pred_labels = torch.argmax(model(rotated_point_clouds), dim=-1, keepdim=False)

        correct_point += pred_labels.eq(labels.data).cpu().sum().item()
        num_point += labels.view([-1, 1]).size()[0]
        preds_labels.append(pred_labels)

    test_accuracy = correct_point / num_point
    print(f"Test accuracy after rotation by {rotation_angle} degrees: {test_accuracy}")

    preds_labels = torch.cat(preds_labels).detach().cpu()

    # Visualize the segmentation results for the first object after rotation
    viz_seg_rot(rotated_point_clouds[0].cpu(), labels[0].cpu(), os.path.join(args.output_dir, f'rotated_gt_seg_{rotation_angle}.gif'), args.device)
    viz_seg_rot(rotated_point_clouds[0].cpu(), preds_labels[0].cpu(), os.path.join(args.output_dir, f'rotated_pred_seg_{rotation_angle}.gif'), args.device)


#Test accuracy after rotation by 30 degrees: 0.7234376012965964