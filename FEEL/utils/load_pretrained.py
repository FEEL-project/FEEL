"""学習済みモデルをロードする関数の定義"""
import yaml
import torch

def load_pretrained_ECO(model_dict, pretrained_model_dict):
    """学習済みモデルをロードする関数
    今回構築したECOは学習済みモデルとレイヤーの順番は同じであることが条件
    レイヤーの名前が異なることに対応できるようにしている
    """

    # 現在のネットワークモデルのパラメータ名
    param_names = []  # パラメータの名前を格納していく
    for name, param in model_dict.items():
        param_names.append(name)

    # 現在のネットワークの情報をコピーして新たなstate_dictを作成
    new_state_dict = model_dict.copy()

    # 新たなstate_dictに学習済みの値を代入
    print("学習済みのパラメータをロードします")
    for index, (key_name, value) in enumerate(pretrained_model_dict.items()):
        name = param_names[index]  # 現在のネットワークでのパラメータ名を取得
        new_state_dict[name] = value  # 値を入れる

        # 何から何にロードされたのかを表示
        print(str(key_name)+"→"+str(name))

    return new_state_dict

def load_config(config_path):
    """設定ファイルを読み込む関数"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config