import numpy as np
import argparse
import os
import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg


def create_parser():
    """Creates a parser for command-line arguments.
    """
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
    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)

    # ------ TO DO: Make Prediction ------
    test_dataloader = get_data_loader(args=args, train=False)

 
    correct_point = 0
    num_point = 0
    preds_labels = []
    for batch in test_dataloader:
        point_clouds, labels = batch
        point_clouds = point_clouds[:, ind].to(args.device)
        labels = labels[:,ind].to(args.device).to(torch.long)

        with torch.no_grad():
            pred_labels = torch.argmax(model(point_clouds), dim=-1, keepdim=False)
        correct_point += pred_labels.eq(labels.data).cpu().sum().item()
        num_point += labels.view([-1,1]).size()[0]

        preds_labels.append(pred_labels)

    test_accuracy = correct_point / num_point
    print(f"test accuracy: {test_accuracy}")
    preds_labels = torch.cat(preds_labels).detach().cpu()


    random_ind = 250
    verts = test_dataloader.dataset.data[random_ind, ind].detach().cpu()
    labels = test_dataloader.dataset.label[random_ind, ind].to(torch.long).detach().cpu()

    correct_point = preds_labels[random_ind].eq(labels.data).cpu().sum().item()
    num_point = labels.view([-1,1]).size()[0]
    accuracy = correct_point / num_point

    # # Visualize Segmentation Result (Pred VS Ground Truth)
    # viz_seg(verts, labels, "{}/gt_seg_3{}.gif".format(args.output_dir, args.exp_name), args.device)
    # viz_seg(verts, preds_labels[random_ind], "{}/pred_seg_3{}.gif".format(args.output_dir, args.exp_name), args.device)
    # Visualize segmentation results for 5 different objects
    num_objects_to_visualize = 5

    # Get indices of 5 random objects
    object_indices = np.random.choice(len(test_dataloader.dataset), num_objects_to_visualize, replace=False)

    for obj_index in object_indices:
        # Get the point clouds and labels for the current object
        point_clouds, labels = test_dataloader.dataset[obj_index]

        # Check if the object has enough points
        if point_clouds.size(1) < args.num_points:
            print(f"Object {obj_index + 1} has fewer points than required. Using all available points.")
            ind = np.arange(point_clouds.size(1))
        else:
            # Randomly select points for each object
            ind = np.random.choice(point_clouds.size(1), args.num_points, replace=False)

        # Get the selected points and labels
        point_clouds = point_clouds[:, ind].unsqueeze(0).to(args.device)

        # Check the dimensionality of labels and perform the indexing accordingly
        if len(labels.size()) == 1:
            labels = labels.to(torch.long)
        elif len(labels.size()) == 2:
            labels = labels[:, ind].unsqueeze(0).to(args.device).to(torch.long)
        else:
            raise ValueError("Unsupported dimensionality for labels")

        with torch.no_grad():
            pred_labels = torch.argmax(model(point_clouds), dim=-1, keepdim=False)

        # Move pred_labels to the same device as labels
        pred_labels = pred_labels.to(args.device)

        # Move labels to the same device as pred_labels
        labels = labels.to(args.device)

        # Calculate accuracy for the current object
        correct_point = pred_labels.eq(labels.data).cpu().sum().item()
        num_point = labels.view([-1, 1]).size()[0]
        accuracy = correct_point / num_point
        # Visualize Segmentation Result (Pred VS Ground Truth)
        viz_seg(point_clouds[0].cpu(), labels[0].cpu(), os.path.join(args.output_dir, f'gt_seg_{obj_index}.gif'), args.device)
        viz_seg(point_clouds[0].cpu(), pred_labels[0].cpu(), os.path.join(args.output_dir, f'pred_seg_{obj_index}.gif'), args.device)

        print(f"Object {obj_index + 1}: test accuracy = {accuracy}")
