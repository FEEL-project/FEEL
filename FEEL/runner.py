import torch
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights

from dataset.video_dataset import load_video_dataset
from model import EnhancedMViT, Elaborator, PFC

def default_mvit(video_dir: str):
    batch_size = 1
    clip_length = 16
    train_loader = load_video_dataset(video_dir, batch_size, clip_length)


    # モデルの準備と推論
    model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)
    model.eval()
    for inputs, labels in train_loader:
        with torch.no_grad():
            outputs = model(inputs)
            # print("Model output shape:", outputs.shape)  # 出力の形状
            max_index = torch.argmax(outputs)
            print(max_index)
            print(labels)

def enhanced_mvit(video_dir: str):
    batch_size = 1
    clip_length = 16
    train_loader = load_video_dataset(video_dir, batch_size, clip_length)


    # モデルの準備と推論
    model = EnhancedMViT(pretrained=True)
    model_pfc = PFC(768, 2, 8).to(device=torch.device("cpu"))
    model.eval()
    model_pfc.eval()
    for inputs, labels in train_loader:
        with torch.no_grad():
            c1,c2,outputs = model(inputs)
            c1_,c2_,outputs_ = model(inputs)
            cat = torch.cat((c2, c2_)).unsqueeze(1)
            print("cat", cat.shape)
            pfc_output = model_pfc(cat)
            print(pfc_output)
            print(pfc_output.shape)
            break

def seeing(video_dir: str, debug=False):
    """
    動画を渡して、評価を出力させる
    """
    batch_size = 2
    clip_length = 16
    train_loader = load_video_dataset(video_dir, batch_size, clip_length)
    hippocampus_params = { # dimension=768, replay_rate=10, replay_iteration=5, size_episode=3
        'dimension': 768,
        'replay_rate': 10,
        'replay_iteration': 5,
        'size_episode': 3
    }
    elaborator = Elaborator(hippocampus_params=hippocampus_params)
    mvit = EnhancedMViT(pretrained=True)
    elaborator.eval()
    mvit.eval()
    evaluation2 = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) # 仮の最終感情評価
    cnt = 0
    for inputs, labels in train_loader:
        if debug:
            print(f"shape of inputs: {inputs.shape}, shape of labels: {labels.shape}")
            print(f"value of labels: {labels}")
        with torch.no_grad():
            _, c2, outputs = mvit(inputs)
            eval_1 = elaborator.subcortical_pathway(c2)
            if debug:
                print(f"shape of c2: {c2.shape}, shape of labels: {labels.shape}")
            events = elaborator.hippocampus.receive(characteristics=c2, evaluation1=labels)
            # events = elaborator.hippocampus.receive(characteristics=c2, evaluation1=eval_1) # eval_1のラベルがない場合(testモード)
            for i in range(len(events)):
                event = events[i]
                if debug and i == 0:
                    print(f"event_id: {event['id']}, shape of characteristics: {event['characteristics'].shape}, shape of evaluation1: {event['evaluation1'].shape}")
                elaborator.hippocampus.save_to_memory(event=event,
                                                        evaluation2=evaluation2)
            if cnt < 10:
                pre_eval = elaborator.prefrontal_cortex(c2)
            else:
                if debug:
                    print(elaborator.hippocampus.num_events)
                if len(events) < batch_size:
                    print(f"WARNING: size of this mini-batch ({len(events)}) is smaller than batch_size ({batch_size})")
                episode = elaborator.hippocampus.generate_episode(events, batch_size=len(events))
                print("episode")
                print(episode)
                pre_eval = elaborator.prefrontal_cortex(episode)
                # break
            cnt += batch_size
            eval_2 = elaborator.controller(eval_1, pre_eval)
            if debug:
                print(f"pre_eval: {pre_eval}")
                print(f"eval_1: {eval_1}")
                print(f"eval_2: {eval_2}")
            print("==============================")


if __name__ == "__main__":
    seeing("./data/small_data/renamed", debug=True)
    # enhanced_mvit("./data/small_data/renamed")
