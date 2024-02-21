import numpy as np
import torch
from models import cls_model
from utils import create_dir, viz_seg_rot
from data_loader import get_data_loader
from argparse import Namespace
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


def visualize_objects_and_failures(args, model, test_data, test_label):
    create_dir(args.output_dir)

    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    rotated_test_data = test_data[:, ind, :]

    # Rotate the entire dataset by 30 degrees
    rotated_test_data = rotate_point_cloud(test_data, 30)

    # Convert to torch tensor
    rotated_test_data = rotated_test_data.numpy().astype(np.float32)
    test_label = torch.from_numpy(test_label).to(args.device)


    # Split the test_data into batches
    test_data_batches = np.array_split(rotated_test_data, range(args.batch_size, rotated_test_data.shape[0], args.batch_size))

    # Convert each batch to a torch tensor and move it to the specified device
    test_data_batches = [torch.from_numpy(batch).float().to(args.device) for batch in test_data_batches]


    # Make Predictions
    all_predictions = []
    with torch.no_grad():
        model.eval()
        for batch in test_data_batches:
            batch = batch.to(args.device)
            predictions = model(batch)
            all_predictions.append(predictions)

    predictions = torch.cat(all_predictions, dim=0)

    # Assuming predictions are class probabilities, get the predicted labels
    _, pred_label = torch.max(predictions, 1)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print("Test accuracy: {}".format(test_accuracy))

    # Visualize Segmentation Result for 5 different objects
    num_objects_to_visualize = 5
    object_indices = np.random.choice(len(test_data), num_objects_to_visualize, replace=False)

    for obj_index in object_indices:
        viz_cls_rot(rotated_test_data[obj_index], test_label[obj_index], "{}/gt_cls_rot{}.gif".format(args.output_dir, obj_index), args.device)
        viz_cls_rot(rotated_test_data[obj_index], pred_label[obj_index], "{}/pred_cls_rot{}.gif".format(args.output_dir, obj_index), args.device)
    
    # Visualize at least 1 failure prediction for each class (chair, vase, and lamp)
    classes_to_visualize = [0, 1, 2]  # Assuming chair, vase, lamp correspond to classes 0, 1, 2
    for class_label in classes_to_visualize:
        # Find the first failure prediction for the specified class
        failure_indices = (test_label == class_label) & (pred_label != class_label)
        if failure_indices.any():
            failure_index = failure_indices.nonzero()[0][0].item()
            viz_cls_rot(test_data[failure_index], test_label[failure_index], "{}/failure_gt_cls_rot{}.gif".format(args.output_dir, class_label), args.device)
            viz_cls_rot(test_data[failure_index], pred_label[failure_index], "{}/failure_pred_cls_rot{}.gif".format(args.output_dir, class_label), args.device)

if __name__ == '__main__':
    # Assuming you have the necessary data loaded for test_data and test_label
    test_data = np.load('./data/cls/data_test.npy')
    test_label = np.load('./data/cls/label_test.npy')

    args = Namespace(
        num_cls_class=6,
        num_points=10000,
        batch_size=32,
        load_checkpoint='model_epoch_0',
        i=0,
        test_data='./data/cls/data_test.npy',
        test_label='./data/cls/label_test.npy',
        output_dir='./output',
        exp_name='exp',
        device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    )

    # Initialize Model for Classification Task
    model = cls_model().to(args.device)

    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("Successfully loaded checkpoint from {}".format(model_path))

    visualize_objects_and_failures(args, model, test_data, test_label)
