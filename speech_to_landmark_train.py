import os
import argparse

from speech_to_landmark import SpeechToLandmarkConfig, SpeechToLandmark


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=15000)
    parser.add_argument("--num_lstms", type=int, default=3)
    parser.add_argument("--tau", type=int, default=18)
    parser.add_argument("--tau_comma", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dataset_dir", type=str, default=r"D:\ObamaWeeklyAddress\tfrecord")
    parser.add_argument("--speech_content_checkpoint", type=str, default=None)
    parser.add_argument("--speaker_aware_checkpoint", type=str, default=None)
    parser.add_argument("--discriminator_checkpoint", type=str, default=None)
    parser.add_argument("--speech_content_model_dir", type=str, default="speech_content_checkpoints")
    parser.add_argument("--speaker_aware_model_dir", type=str, default="speaker_aware_checkpoints")
    parser.add_argument("--discriminator_model_dir", type=str, default="discriminator_checkpoints")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--is_2d", type=str2bool, default=True)
    parser.add_argument("--summary_dir", type=str, default="summary")
    parser.add_argument("--summary_frequency", type=int, default=10)
    parser.add_argument("--save_network_frequency", type=int, default=100)
    parser.add_argument("--is_training", type=str2bool, default=True)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--per_process_gpu_memory_fraction", type=float, default=0.6)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--learning_rate_decay_type", type=str, default="constant")
    parser.add_argument("--decay_steps", type=int, default=None)
    parser.add_argument("--decay_rate", type=float, default=0.9)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--lambda_c", type=float, default=1.0)
    parser.add_argument("--lambda_s", type=float, default=1.0)
    parser.add_argument("--miu_s", type=float, default=1e-3)
    parser.add_argument("--use_speaker_aware", type=str2bool, default=True)
    parser.add_argument("--is_loadmodel", type=str2bool, default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
