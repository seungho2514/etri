import soundata
import os
import urllib.request
# 데이터 저장할 루트 폴더
DATA_ROOT = "/data/ACoM"

def download_dataset(dataset_name, output_dir):
    dataset = soundata.initialize(dataset_name, data_home=output_dir)
    dataset.download()
    dataset.validate()


if __name__ == "__main__":
    os.makedirs(DATA_ROOT, exist_ok=True)

    #download_dataset('esc50', os.path.join(DATA_ROOT, "ESC-50"))
    download_dataset('urbansound8k', os.path.join(DATA_ROOT, "UrbanSound8K"))