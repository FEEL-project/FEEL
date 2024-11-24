import torch
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights
import logging

from dataset.video_dataset import load_video_dataset
from model import EnhancedMViT, PFC, Hippocampus, HippocampusRefactored, SubcorticalPathway, EvalController
from utils import timeit

BATCH_SIZE = 2
CLIP_LENGTH = 16
DIM_CHARACTERISTICS = 768
SIZE_EPISODE = 3
DEVICE = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

@timeit
def main(video_path: str):
    train_loader = load_video_dataset(video_path, "annotation/params_trainval.csv", BATCH_SIZE, CLIP_LENGTH)
    model_mvit = EnhancedMViT(pretrained=True).to(device=DEVICE)
    model_pfc = PFC(DIM_CHARACTERISTICS, SIZE_EPISODE, 8).to(device=DEVICE)
    model_hippocampus = HippocampusRefactored(
        DIM_CHARACTERISTICS,
        SIZE_EPISODE,
        replay_rate=10,
        episode_per_replay=5,
        min_event_for_episode=5,
    )

    model_subcortical_pathway = SubcorticalPathway()
    model_controller = EvalController()
    model_mvit.eval()
    model_pfc.eval()
    cnt = 0
    eval2 = torch.Tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    for inputs, labels in train_loader:
        logging.debug(f"inputs: {inputs.shape}, labels: {labels.shape}")
        logging.debug(f"labels: {labels}")
        with torch.no_grad():
            # Calculates eval1 (Intuitive emotional response)
            _, characteristics, _ = model_mvit(inputs)
            eval1 = model_subcortical_pathway(characteristics) #(1)
            events = model_hippocampus.receive(characteristics, labels)
            for i in range(len(events)):
                event = events[i]
                if i==0:
                    logging.debug(f"event_id: {event.id}, characteristics: {event.characteristics.shape}, evaluation1: {event.eval1.shape}")
                model_hippocampus.save_to_memory(event=event, eval1=labels, eval2=eval2)
            if cnt < model_hippocampus.min_event_for_episode: # If not enough events to generate episode, use the memory itself as episode
                # Zero padding to (SIZE_EPISODE, BATCH_SIZE, DIM_CHARACTERISTICS)
                pre_eval = torch.zeros((SIZE_EPISODE, BATCH_SIZE, DIM_CHARACTERISTICS))
                pre_eval[0, :, :] = characteristics
                pre_eval = model_pfc(pre_eval)
            else: # Generate episode and calculate pre_eval
                logging.debug(f"Number of memories: {len(model_hippocampus)}")
                episode = model_hippocampus.generate_episodes_batch(events=events)
                logging.debug(f"episode: {episode.shape}")
                pre_eval = model_pfc(episode.transpose(0, 1))
            eval2 = model_controller(eval1, pre_eval)
            logging.debug(f"eval2: {eval2.shape}")
            logging.info(
                f"cnt: {cnt}\neval1: {eval1}\npre_eval: {pre_eval}\neval2: {eval2}\n"
                )
            cnt += BATCH_SIZE

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    if DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    main("./data/small_data/trainval")
    # enhanced_mvit("./data/small_data/renamed")
