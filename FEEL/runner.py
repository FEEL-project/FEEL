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

def seeing(video_dir: str):
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
        print("inputs")
        print(inputs.shape)
        print("labels")
        print(labels)
        with torch.no_grad():
            c1, c2, outputs = mvit(inputs)
            # print(f"shape of c2: {c2.shape}, shape of labels: {labels.shape}")
            events = elaborator.hippocampus.receive(characteristics=c2, evaluation1=labels) ### issue: mini-batchに対応していない->解消
            # print("event")
            # print(f"length of event['id']: {len(event['id'])}, shape of event['characteristics']: {event['characteristics'].shape},shape of event['evaluation1']: {event['evaluation1'].shape}")
            if cnt < 10:
                for i in range(len(events)):
                    event = events[i]
                    elaborator.hippocampus.save_to_memory(event=event,
                                                          evaluation2=evaluation2)
            else:
                ### issue: hippocampusに十分な数のeventがないため、episodeが生成されない
                print(elaborator.hippocampus.num_events)
                episode = elaborator.hippocampus.generate_episode(events, batch_size=batch_size)
                print("episode")
                print(episode)
                break
            cnt += batch_size
            pre_eval = elaborator.prefrontal_cortex(c2)
            # pre_eval = elaborator.prefrontal_cortex(episode)
            eval_1 = elaborator.subcortical_pathway(c2)
            eval_2 = elaborator.controller(eval_1, pre_eval)
            # print("pre_eval")
            # print(pre_eval)
            # print("eval_1")
            # print(eval_1)
            # print("eval_2")
            # print(eval_2)


if __name__ == "__main__":
    seeing("./data/small_data/renamed")
    # enhanced_mvit("./data/small_data/renamed")
