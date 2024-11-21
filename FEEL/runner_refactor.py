import torch
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights

from dataset.video_dataset import load_video_dataset
from model import EnhancedMViT, PFC, Hippocampus, HippocampusRefactored, SubcorticalPathway, EvalController
from utils import timeit

@timeit
def main(video_path: str, use_old: bool = False):
    BATCH_SIZE = 2
    CLIP_LENGTH = 16
    DIM_CHARACTERISTICS = 768
    SIZE_EPISODE = 3
    global DEVICE
    DEVICE = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    train_loader = load_video_dataset(video_path, "annotation/params_trainval.csv", BATCH_SIZE, CLIP_LENGTH)
    
    model_mvit = EnhancedMViT(pretrained=True).to(device=DEVICE)
    model_pfc = PFC(DIM_CHARACTERISTICS, SIZE_EPISODE, 8).to(device=DEVICE)
    if not use_old:
        model_hippocampus = HippocampusRefactored(
            DIM_CHARACTERISTICS,
            SIZE_EPISODE,
            replay_rate=10,
            episode_per_replay=5,
            min_event_for_episode=5,
        )
    else:
        model_hippocampus = Hippocampus(
            DIM_CHARACTERISTICS,
            size_episode=SIZE_EPISODE,
            replay_rate=10,
            replay_iteration=5,
            minimal_to_generate=5
        )
    model_subcortical_pathway = SubcorticalPathway()
    model_controller = EvalController()
    model_mvit.eval()
    model_pfc.eval()
    cnt = 0
    eval2 = torch.Tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    for inputs, labels in train_loader:
        print("====================================")
        print("inputs", inputs.shape)
        print("labels", labels)
        with torch.no_grad():
            _, characteristics, _ = model_mvit(inputs)
            print("characteristics", characteristics.shape)
            events = model_hippocampus.receive(characteristics, labels)
            print("events", len(events))
            if cnt < 5:
                for i in range(len(events)):
                    event = events[i]
                    if use_old:
                        model_hippocampus.save_to_memory(event=event, evaluation2=eval2)
                    else:
                        model_hippocampus.save_to_memory(event=event, eval2=eval2)
            else:
                if use_old:
                    episode = model_hippocampus.generate_episode(events, batch_size=BATCH_SIZE)
                else:
                    episode = model_hippocampus.generate_episodes_batch(events=events)
                print("episode", episode)
                pre_eval = model_pfc(episode.transpose(0, 1))
                print("pre_eval", pre_eval.shape)
                eval1 = model_subcortical_pathway(characteristics)
                print("eval1", eval1.shape)
                eval2 = model_controller(eval1, pre_eval)
                print("eval2", eval2)
                break
            cnt += BATCH_SIZE

if __name__ == "__main__":
    main("./data/small_data/trainval", use_old=False)
    # enhanced_mvit("./data/small_data/renamed")
